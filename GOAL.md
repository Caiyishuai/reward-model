# AutoRM Development Goals

## 硬约束（不可违背）

以下四条为**项目目标定义**，实验与代码优化均须满足；违反任一条即不算完成核心命题。

| # | 约束 | 说明 |
|---|------|------|
| **D1** | **RM 训练数据固定** | Reward Model **必须且只能**用提供的 **20 条成功 + 20 条失败** 演示及其经 auto-labelling 得到的标签来训练（不得换更大演示集、不得混入额外人工 dense reward 作为 RM 监督，除非明确记为对照实验）。 |
| **D2** | **SAC 必须用 RM 训练** | 主实验路径下 policy 的 RL 更新 **必须以 RM 产生的 reward 为训练信号**；核心验证为 **`--env-reward-scale 0.0`**（零环境 shaping），与 D3 的 eval 配套。 |
| **D3** | **Eval 只看环境指标** | 最终好坏 **只认 ManiSkill3 环境给出的 episode 统计**：**成功率**（如 `success_once` / `success_at_end`）与 **return**；eval **不使用 RM**，与训练 reward 解耦。 |
| **D4** | **全程 GPU 训练 NN** | RM 训练、RM 推理/relabelling、SAC 训练涉及的神经网络均在 GPU 上执行（见下文「GPU 要求」）。 |

**与 D2 的关系：** 允许存在 **对照实验**（例如 `env_reward_scale>0`）用于分析，但**论文级/命题级结论**必须基于 D1+D2+D3 的主线；对照不能替代主线。

---

## Project Mission

**AutoRM 的核心命题：** 机器人操作任务里，手工设计 reward 代价高且难以迁移。能否只用
**20 success + 20 failure 的二值演示**（不标任何 per-step reward），通过自动化 labelling
生成足够质量的 dense reward signal，进而训练出有效的 policy？

完整验证链路如下：

```
提供的 20 success + 20 fail demos  (仅二值 episode label，无人工 per-step reward)
        ↓  auto labelling  (HMM + HGBC + KNN，label/auto_label.py)
  per-step reward labels   (auto_processed/)
        ↓  训练 Reward Model  (DINOv2 + Temporal Transformer)，监督仅来自上述数据管线
  RM checkpoint
        ↓  纯 RM reward 驱动 SAC（env_reward_scale=0，满足 D2）
  policy 在 env 中的 eval：成功率 + return  ← 满足 D3，唯一验收指标
```

**关键约束：** 最终验证必须是 **纯 RM reward 驱动**（零 env reward），才能证明
auto labelling → RM 这条链路的质量可以独立替代手工 reward。混入 env reward 只能证明
「RM 有辅助作用」，不能证明核心命题。

---

## Current Iteration: SAC + RM Pipeline Verification

### Core Objectives

| # | Goal | Status | Notes |
|---|------|--------|-------|
| 0 | 分析 ManiSkill3 官方 SAC（RGB+state） | ✅ Complete | 对比 ManiSkill3、LeanRL、CleanRL、v1/v2；v2 对齐官方 sac_rgbd.py。 |
| 1 | 学习 LeanRL → 构建自有 SAC v2 | ✅ Complete | `train()` + `run_eval()` 可抽离；`torch.compile` + AMP 加速。 |
| 2 | 纯 env reward SAC 在 PushCube 验证能训通 | ✅ Verified | 200k: success=94%, return=36.55（40% 进度时）。baseline 确立。 |
| 3 | 验证 auto labelling + 训练 Reward Model | ✅ Complete | `rm_pushcube_epoch_55_val_0.8286.pt`，val_rank_acc=82.86%。 |
| 4a | RM + env reward 混合训练（对照组） | ✅ Done | 200k, 189 SPS. eval success=87.5%, return=39.81. α=0.5（混合，不能证明核心命题）。 |
| **4b** | **纯 RM reward 驱动 SAC（零 env reward）** | ✅ **Done（未达 baseline）** | 200k，`/tmp/rm_only_200k.log`，~1031s，**194 SPS**。TensorBoard：`eval/success_once` **last=0%，max=25%**；`eval/return` **last≈2.58，max≈11.08**。远低于 Step 2（~94%），说明 **当前 RM 作为唯一 reward 尚不足以复现 env-reward 性能**，需继续迭代 RM / reward 归一化 / 训练预算。 |

### 实验设计说明

| 实验 | training reward | eval 指标来源 | 能证明什么 |
|------|----------------|--------------|-----------|
| Step 2 | 纯 `env_reward` | env episode stats | baseline：有 GT reward 能学多好 |
| Step 4a（已完成） | `env_reward + 0.5 * rm_reward` | env episode stats | RM 有辅助提升；无法排除 env_reward 贡献 |
| **Step 4b（已跑）** | **纯 `rm_reward`（env_reward=0）** | **env episode stats** | **本轮未验证成功**（eval success 峰值 25%）；命题仍开放，需改进 RM 或学习动态再跑 |

> **注：eval 的 `success_once` 和 `return` 始终来自 env 的 episode 统计
> （`infos["final_info"]["episode"]`），与 RM reward 无关。eval 阶段完全不使用 RM。**

---

## 持续优化：Auto-Experiment（`.cursor/skills/21-auto-experiment`）

在 **不违反 D1–D4** 的前提下，用 **自主实验闭环** 迭代 RM 管线、reward 归一化、SAC 超参等，目标拉高 **D3 的 eval 成功率与 return**。

**指标（skill 要求 ≥2 个：主指标 + 护栏）：**

| 角色 | 指标 | 方向 | 说明 |
|------|------|------|------|
| **Primary** | `eval/success_once`（或任务约定的 success） | 越高越好 | 对齐核心命题；从 TensorBoard 或 `run_eval` 日志读取。 |
| **Guard** | `eval/return` | 不应长期低于荒谬区间 | 防止策略「投机」单一指标；可与 Step 2 env-only baseline 的 return 量级对照。 |
| **Guard（可选）** | 训练墙钟 / SPS | 不应无限变慢 | 超时按 skill：超过 baseline 约 2× 则判 timeout，避免无限试错。 |

**修改面（典型）：** `train.py`（RM）、`label/auto_label.py` 与配置（labelling）、`sim/sac_train_v2.py` 中与 **RM relabel / 归一化 / `env_reward_scale`** 相关的参数路径；**禁止**用额外演示数据满足 D1。

**流程摘要：** Reconnaissance → 指标与 sanity check → baseline（当前 4b 曲线可作为锚）→ 假设 → 单次改动 → 完整 eval → ledger（JSONL）→ keep/discard → 汇总。详见 skill 内 **Iron Law** 与 **Workflow**。

**当前会话：** Ledger `experiments/ledger_sac_rm.jsonl`。实验 1（**已结束**）：`rm-only-no-norm`，`eval/success_once` **max=0.25 / last=0**（与 4b 相同），`eval/return` **max=9.30 / last=3.21**；相对 baseline **return max 下降**，记为 **discard**。下一假设可试：`rm_clip`、`--no-rm-uncertainty-weight`、`relabel_interval` 等（仍满足 D1–D4）。

---

## GPU 要求（强制）

> ⚠️ **所有神经网络训练均必须使用 GPU。** 以下三个阶段无一例外：

| 阶段 | 涉及的神经网络 | GPU 必要性 |
|------|--------------|-----------|
| **Reward Model 训练** (`train.py`) | DINOv2 backbone + Temporal Transformer + Ensemble Heads | DINOv2 ViT-S/16 参数量 ~21M，CPU 不可接受 |
| **RM 推理 / relabelling** (`sim/sac_train_v2.py` RMRelabeler) | 同上（backbone batch inference） | 每次 relabel 要对 buffer 批量推理，无 GPU 极慢 |
| **SAC 训练** (`sim/sac_train_v2.py`) | PlainConv CNN + Actor + Critic（GPU sim + GPU replay buffer） | `sim_backend=gpu`、`buffer_device=cuda` 硬要求，代码有强制检查 |

**硬约束（`sim/sac_train_v2.py` 中已写死）：**
- `cuda=True`（默认）→ CUDA 可用性检查失败直接 `RuntimeError`，无 CPU 回退。
- `sim_backend=gpu`：ManiSkill3 GPU 并行仿真。
- `buffer_device=cuda`：replay buffer 在 GPU 上。

**当前环境：** RTX 4090 24GB，驱动 570（CUDA 12.8），PyTorch **`2.6.0+cu124`**（锁定在 `pyproject.toml`）。

---

## SAC Implementation Summary

**Primary file:** `sim/sac_train_v2.py`

Based on ManiSkill3 `sac_rgbd.py` + LeanRL patterns:
- PlainConv CNN shared between Actor and Critic
- GPU replay buffer via DictArray
- GPU-vectorized parallel environments (ManiSkillVectorEnv)
- RM relabeling：CPU 侧存储 obs，后台线程异步 batch 推理，写回 GPU reward buffer

**Acceleration features:**
- `--compile` — torch.compile on all networks (~1.5–2x speedup)
- `--amp` — automatic mixed precision (fp16), only active when `device=cuda`
- `--num-envs 32` — more parallel environments
- `--training-freq 128` — more env steps per iteration

**Speed benchmarks (200k PushCube, 64×64 RGB, RTX 4090):**

| Config | SPS | 200k ETA |
|--------|-----|----------|
| 32 envs, compile+AMP, rm_alpha=0.5 | ~189 | ~17.5 min |
| 32 envs, compile+AMP, no RM | ~65 it/s | ~30 min |

---

## Simulation Stack (ManiSkill3)

- PyPI 包名 **`mani-skill`**，**主版本 3** = **ManiSkill3**（不是 2.x）。
- **Python：** 使用 **3.10**（仓库有 `.python-version`）；`toppra`（ManiSkill3 依赖）在 Linux 上只有 cp310 wheel，3.11 会装失败。
- **安装：** `uv sync --extra sim`（`gymnasium` 由 ManiSkill3 的 pin 带入，不要在 `sim` extra 里再声明 `gymnasium>=1.x`）。
- **PyTorch：** `pyproject.toml` 里固定为 `torch==2.6.0 + torchvision==0.21.0`，来源 `https://download.pytorch.org/whl/cu124`（cu124 匹配 CUDA 12.x 驱动，不用 cu130）。

---

## Phase 4b — 纯 RM reward 驱动 SAC（✅ 已跑完一轮）

**本轮结果（TensorBoard，`runs/PushCube-v1__sac_pushcube_200k_rm_only__1__1776441324/`）：**
- `eval/success_once`：**max 0.25，last 0.00**（8 个 eval env 上均值）
- `eval/return`：**max ≈11.08，last ≈2.58**
- 训练：`200064` env steps，`~1031s`，**~194 SPS**；`final_ckpt.pt` 已写入上述目录。

**监控长跑日志（tqdm 用 `\r` 刷新，勿直接 `tail` 一行）：**
```bash
watch -n 30 "perl -pe 's/\\r/\\n/g' /tmp/rm_only_200k.log | grep 'success:' | tail -3"
```

```bash
# 通过脚本一键启动（rm-only 为默认 preset）
uv run --extra sim python scripts/run_pushcube_sac_v2_rm.py --preset rm-only

# 等价完整命令（env_reward 完全归零）：
uv run --extra sim python -m sim.sac_train_v2 \
    --env-id PushCube-v1 \
    --robot-uids panda_wristcam \
    --num-envs 32 --num-eval-envs 8 \
    --total-timesteps 200000 \
    --buffer-size 100000 \
    --compile --amp \
    --camera-width 64 --camera-height 64 \
    --gamma 0.8 --tau 0.01 \
    --rm-checkpoint checkpoints/auto_pushcube/rm_pushcube_epoch_55_val_0.8286.pt \
    --rm-alpha 1.0 \
    --env-reward-scale 0.0 \
    --exp-name sac_pushcube_200k_rm_only
```

> **验收标准：** eval `success_once` 接近 step 2（纯 env reward baseline ~94%），
> 说明 auto labelling → RM 链路可以独立替代手工 reward。**本轮未满足**，需后续实验（RM、归一化、步数等）。

---

## Knowledge Base References

- **DB-007:** Vision SAC 5x slowdown from CPU replay buffer (solved)
- eval_freq semantics fixed to iteration-based counting
- torch cu130 与 CUDA 12.x 驱动不兼容 → 改用 cu124（已解决）
- uv sync 多进程并发 venv 锁 → 保持同一时刻只跑一个 uv sync（已记录）
- **持续优化：** `.cursor/skills/21-auto-experiment/SKILL.md`（指标、ledger、`exp/` 分支生命周期）
