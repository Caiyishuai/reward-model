"""RM-guided PPO training in ManiSkill3 environments.

Trains a PPO policy where the reward signal is a mix of the environment
sparse/dense reward and the learned Reward Model score:

    ``shaped_reward = env_reward + alpha * rm_reward``

The RM is used in inference-only mode; its gradients do not flow into the
policy.  RM inference is amortised by running every ``rm_every_k`` steps
(non-RM steps use environment reward only, no interpolation).

Architecture matches the PPO Agent in ``1_collect_dp_data.py`` so that
checkpoints are interchangeable.

Usage::

    python -m sim.ppo_train \\
        --task pushcube \\
        --rm_checkpoint checkpoints/auto_pushcube/rm_pushcube_epoch_55_val_0.8286.pt \\
        --total_timesteps 500000 \\
        --reward_shaping_weight 0.5
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from checkpoint_io import load_rl_checkpoint, save_rl_checkpoint
from reward_model import RewardModel
from sim.agents import PPOAgent
from sim.env_factory import make_env
from sim.obs_adapter import ObsAdapter, extract_full_state
from sim.task_configs import get_sim_config

# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    task: str = "pushcube"
    rm_checkpoint: str | None = None
    ppo_init_checkpoint: str | None = None
    reward_shaping_weight: float = 0.5
    rm_every_k: int = 1

    total_timesteps: int = 500_000
    num_envs: int = 1
    num_steps: int = 200
    num_minibatches: int = 4
    update_epochs: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    eval_freq: int = 10_000
    eval_episodes: int = 20
    save_freq: int = 50_000
    output_dir: str = "runs"
    seed: int = 42
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


def evaluate_policy(env, agent: PPOAgent, device: torch.device, n_episodes: int, max_steps: int) -> dict[str, float]:
    """Evaluate policy success rate without RM (pure env metric)."""
    agent.eval()
    successes = 0
    total_rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        for _ in range(max_steps):
            state = obs["state"] if isinstance(obs, dict) else obs
            t = (
                state.float().to(device)
                if isinstance(state, torch.Tensor)
                else torch.from_numpy(np.asarray(state)).float().to(device)
            )
            with torch.no_grad():
                action = agent.actor_mean(t)
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            r = reward.item() if hasattr(reward, "item") else float(reward)
            ep_reward += r
            done = terminated or truncated
            if hasattr(done, "any"):
                done = bool(done.any())
            if done:
                success = info.get("success", False)
                if hasattr(success, "any"):
                    success = bool(success.any())
                if success:
                    successes += 1
                break
        total_rewards.append(ep_reward)

    agent.train()
    return {
        "success_rate": successes / max(n_episodes, 1),
        "mean_reward": float(np.mean(total_rewards)),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def _build_ppo_meta(
    cfg: PPOConfig,
    sim_cfg,
    global_step: int,
    *,
    stage: str,
) -> dict[str, object]:
    """Return the meta dict embedded into every PPO checkpoint."""
    return {
        "agent": "ppo",
        "stage": stage,
        "task": cfg.task,
        "env_id": sim_cfg.env_id,
        "total_timesteps": cfg.total_timesteps,
        "global_step": int(global_step),
        "num_envs": cfg.num_envs,
        "num_steps": cfg.num_steps,
        "rm_checkpoint": cfg.rm_checkpoint,
        "rm_alpha": cfg.reward_shaping_weight,
        "rm_every_k": cfg.rm_every_k,
        "ppo_init_checkpoint": cfg.ppo_init_checkpoint,
        "seed": cfg.seed,
    }


def train(cfg: PPOConfig) -> None:
    """Run RM-guided PPO training."""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    sim_cfg = get_sim_config(cfg.task)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)  # noqa: NPY002 — kept for legacy torch/gym code paths; PPO shuffling uses default_rng below.
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    run_name = f"ppo_{cfg.task}_{cfg.seed}_{int(time.time())}"
    run_dir = Path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PPO] Task: {sim_cfg.env_id}")
    print(f"[PPO] RM: {cfg.rm_checkpoint or 'none (env reward only)'}")
    print(f"[PPO] Shaping weight alpha: {cfg.reward_shaping_weight}")
    print(f"[PPO] Output: {run_dir}")

    if cfg.num_envs != 1:
        raise NotImplementedError(
            f"PPO currently supports num_envs=1 only (got {cfg.num_envs}). "
            "Buffers and obs extraction assume single-env layout."
        )

    env = make_env(sim_cfg, num_envs=cfg.num_envs)

    obs_test, _ = env.reset()
    full_state = extract_full_state(obs_test)
    n_obs = len(full_state)
    n_act = math.prod(env.action_space.shape)
    print(f"[PPO] obs_dim={n_obs}, act_dim={n_act}")

    agent = PPOAgent(n_obs, n_act, device=device)
    if cfg.ppo_init_checkpoint:
        state, ckpt_meta = load_rl_checkpoint(cfg.ppo_init_checkpoint, map_location=device)
        # Accept both unified artifact ({agent: {...}}) and legacy flat state_dict.
        agent_sd = state.get("agent", state) if isinstance(state, dict) else state
        agent.load_state_dict(agent_sd)
        if ckpt_meta:
            print(f"[PPO] Warm-started from {cfg.ppo_init_checkpoint} (meta: {ckpt_meta})")
        else:
            print(f"[PPO] Warm-started from {cfg.ppo_init_checkpoint}")

    optimizer = optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    rm: RewardModel | None = None
    adapter: ObsAdapter | None = None
    alpha = cfg.reward_shaping_weight
    if cfg.rm_checkpoint:
        rm = RewardModel.load(cfg.rm_checkpoint, device=str(device))
        adapter = ObsAdapter(sim_cfg, state_windows=rm.state_windows, device=device)
        print(f"[PPO] RM loaded (state_windows={rm.state_windows})")

    eval_env = env  # reuse same env to avoid SAPIEN multi-instance segfault

    batch_size = cfg.num_envs * cfg.num_steps
    minibatch_size = batch_size // cfg.num_minibatches
    num_updates = cfg.total_timesteps // batch_size

    obs_buf = torch.zeros(cfg.num_steps, n_obs, device=device)
    act_buf = torch.zeros(cfg.num_steps, n_act, device=device)
    logprob_buf = torch.zeros(cfg.num_steps, device=device)
    reward_buf = torch.zeros(cfg.num_steps, device=device)
    done_buf = torch.zeros(cfg.num_steps, device=device)
    value_buf = torch.zeros(cfg.num_steps, device=device)

    global_step = 0
    best_success_rate = 0.0
    start_time = time.time()

    obs, _ = env.reset()
    if adapter is not None:
        adapter.reset(env, obs)

    next_state = extract_full_state(obs)
    next_state_t = torch.from_numpy(next_state).float().to(device)
    next_done = torch.zeros(1, device=device)

    print(f"\n[PPO] Training for {num_updates} updates ({cfg.total_timesteps} timesteps)\n")

    for update in range(1, num_updates + 1):
        frac = 1.0 - (update - 1) / num_updates
        lr_now = frac * cfg.lr
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        for step in range(cfg.num_steps):
            global_step += cfg.num_envs
            obs_buf[step] = next_state_t
            done_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_state_t.unsqueeze(0))
                action = action.squeeze(0)
                logprob = logprob.squeeze(0)
                value = value.squeeze(0)

            act_buf[step] = action
            logprob_buf[step] = logprob
            value_buf[step] = value

            obs, env_reward, terminated, truncated, info = env.step(action.cpu().numpy())

            if adapter is not None:
                adapter.step(env, obs)

            r_env = env_reward.item() if hasattr(env_reward, "item") else float(env_reward)

            r_rm = 0.0
            if rm is not None and adapter is not None and step % cfg.rm_every_k == 0:
                images, proprio = adapter.get_rm_inputs()
                r_rm = rm.get_reward(images, proprio).item()

            reward_buf[step] = r_env + alpha * r_rm

            done = terminated or truncated
            if hasattr(done, "any"):
                done = bool(done.any())

            if done:
                obs, _ = env.reset()
                if adapter is not None:
                    adapter.reset(env, obs)

            next_state = extract_full_state(obs)
            next_state_t = torch.from_numpy(next_state).float().to(device)
            next_done = torch.tensor([1.0 if done else 0.0], device=device)

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_state_t.unsqueeze(0)).squeeze()
            advantages = torch.zeros(cfg.num_steps, device=device)
            lastgaelam = 0.0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done.squeeze()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - done_buf[t + 1]
                    nextvalues = value_buf[t + 1]
                delta = reward_buf[t] + cfg.gamma * nextvalues * nextnonterminal - value_buf[t]
                advantages[t] = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + value_buf

        b_obs = obs_buf.reshape(batch_size, -1)
        b_actions = act_buf.reshape(batch_size, -1)
        b_logprobs = logprob_buf.reshape(batch_size)
        b_advantages = advantages.reshape(batch_size)
        b_returns = returns.reshape(batch_size)
        b_values = value_buf.reshape(batch_size)

        b_inds = np.arange(batch_size)
        clipfracs = []

        for _ in range(cfg.update_epochs):
            rng.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

        elapsed = time.time() - start_time
        sps = int(global_step / elapsed)
        if update % 5 == 0 or update == 1:
            print(
                f"Update {update:4d}/{num_updates} | "
                f"step={global_step:7d} | "
                f"pg={pg_loss.item():.4f} vf={v_loss.item():.4f} ent={entropy_loss.item():.4f} | "
                f"kl={approx_kl.item():.4f} | lr={lr_now:.2e} | SPS={sps}"
            )

        if global_step % cfg.eval_freq < batch_size:
            metrics = evaluate_policy(eval_env, agent, device, cfg.eval_episodes, sim_cfg.max_episode_steps)
            print(
                f"  [EVAL] step={global_step}  success_rate={metrics['success_rate'] * 100:.1f}%  "
                f"mean_reward={metrics['mean_reward']:.3f}"
            )
            if metrics["success_rate"] > best_success_rate:
                best_success_rate = metrics["success_rate"]
                best_path = run_dir / "best.pt"
                save_rl_checkpoint(
                    best_path,
                    state={"agent": agent.state_dict()},
                    meta=_build_ppo_meta(cfg, sim_cfg, global_step, stage="best"),
                )
                print(f"  [EVAL] New best! Saved to {best_path}")

        if global_step % cfg.save_freq < batch_size:
            ckpt_path = run_dir / f"ckpt_{global_step}.pt"
            save_rl_checkpoint(
                ckpt_path,
                state={"agent": agent.state_dict()},
                meta=_build_ppo_meta(cfg, sim_cfg, global_step, stage="checkpoint"),
            )

    env.close()

    final_path = run_dir / "final.pt"
    save_rl_checkpoint(
        final_path,
        state={"agent": agent.state_dict()},
        meta=_build_ppo_meta(cfg, sim_cfg, cfg.total_timesteps, stage="final"),
    )
    print(f"\n[PPO] Training complete. Best success rate: {best_success_rate * 100:.1f}%")
    print(f"[PPO] Final checkpoint: {final_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="RM-guided PPO training in ManiSkill3")
    parser.add_argument("--task", type=str, default="pushcube")
    parser.add_argument("--rm_checkpoint", type=str, default=None)
    parser.add_argument("--ppo_init_checkpoint", type=str, default=None)
    parser.add_argument("--reward_shaping_weight", type=float, default=0.5)
    parser.add_argument("--rm_every_k", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--save_freq", type=int, default=50_000)
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    sim_cfg = get_sim_config(args.task)
    rm_ckpt = args.rm_checkpoint or sim_cfg.rm_checkpoint

    cfg = PPOConfig(
        task=args.task,
        rm_checkpoint=rm_ckpt,
        ppo_init_checkpoint=args.ppo_init_checkpoint,
        reward_shaping_weight=args.reward_shaping_weight,
        rm_every_k=args.rm_every_k,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        lr=args.lr,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_freq=args.save_freq,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
    )
    train(cfg)


if __name__ == "__main__":
    main()
