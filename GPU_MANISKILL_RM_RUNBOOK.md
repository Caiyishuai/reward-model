# ManiSkill 四任务 RM GPU 运行手册

本文记录在 Linux/NVIDIA GPU 机器上为以下四个任务采集数据、生成自动标签、
训练 Reward Model checkpoint 并评估质量的完整步骤：

- `PushCube-v1` → `pushcube`
- `PokeCube-v1` → `pokecube`
- `PlaceSphere-v1` → `placesphere`
- `StackCube-v1` → `stackcube`

目标数据协议固定为：**每任务 20 success + 20 fail episode、双相机 RGB、
17-D `eef+joint` state、7-D `pd_ee_delta_pose` action**。不要用现有
`pkl_data_ppo` 旧文件替代：其中存在 4-D action 或 vector-env batch 未按
episode 环境切分的问题。

## 1. 最终应得到什么

每个任务应产生：

```text
reward-model/
├── data/<task>/
│   ├── success_raw.pkl
│   ├── fail_raw.pkl
│   ├── collection_meta.json
│   ├── episodes/{success,fail}/episode_*.pkl
│   └── auto_processed/
│       ├── success_lerobot.pkl
│       ├── fail_lerobot.pkl
│       └── <task>_dashboard.png
├── checkpoints/auto_<task>/
│   ├── best.pt
│   ├── best.json
│   └── rm_<task>_epoch_*_val_*.pt
└── eval_results/maniskill_rm_quality.json
```

`best.pt` 是后续仿真/SERL 使用的稳定 checkpoint 路径。

## 2. 实验约束与验收标准

运行中不得放宽以下约束：

1. success/fail 各至少 20 条完整 episode。
2. train/validation 按 episode 切分，默认 `seed=42`、`split_ratio=0.9`。
3. RM 不得看到 validation episode 的训练样本。
4. action 必须为 7-D `pd_ee_delta_pose`。
5. state 必须为 17-D：
   `eef_xyz(3)+eef_axis_angle(3)+fingers(2)+joint_qpos(9)`。
6. 相机必须包含 `hand_camera` 和 `base_camera`，图像为 `(H,W,3) uint8`。
7. ManiSkill dense reward 单独存为 `env_rewards`；自动标签写入 `rewards`，
   两者不能互相覆盖。
8. RM 离线指标只是信号诊断。最终有效性仍需 RM-only policy 的环境
   `success_once` / success-AUC 多种子实验。

## 3. GPU 机器准备

假设两个仓库处于同级目录：

```bash
export WORK_ROOT=/path/to/mywork
export MANISKILL_ROOT="$WORK_ROOT/maniskill-ws"
export REWARD_MODEL_ROOT="$WORK_ROOT/reward-model"
```

记录代码版本，避免结果无法追踪：

```bash
git -C "$MANISKILL_ROOT" rev-parse HEAD
git -C "$REWARD_MODEL_ROOT" rev-parse HEAD
git -C "$MANISKILL_ROOT" status --short
git -C "$REWARD_MODEL_ROOT" status --short
```

检查 GPU：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

预期最后一项为 `True`。建议至少 24 GB 显存；多 GPU 可以缩短四任务 RM
训练，但数据采集仍建议单环境运行，以保证 episode 边界正确。

### 3.1 ManiSkill 环境

优先复用已经能运行 `rl/1_ppo_fast.py` 的环境。若新建环境：

```bash
cd "$MANISKILL_ROOT"
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install "mani-skill>=3.0.0b22,<4" tensordict tensorboard tyro tqdm
```

验证四个环境和 7-D control mode：

```bash
python - <<'PY'
import gymnasium as gym
import mani_skill.envs

for env_id in ("PushCube-v1", "PokeCube-v1", "PlaceSphere-v1", "StackCube-v1"):
    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="state",
        sim_backend="physx_cuda",
        control_mode="pd_ee_delta_pose",
    )
    obs, _ = env.reset(seed=42)
    print(env_id, "obs", obs.shape, "action", env.action_space.shape)
    env.close()
PY
```

若 `PlaceSphere-v1` 未注册，先检查当前 ManiSkill 版本；不要把其他任务
重命名为 PlaceSphere 代替。

### 3.2 Rsync 环境

```bash
cd "$REWARD_MODEL_ROOT"
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[sim]"
```

确认 DINO/torch 可用：

```bash
python -c "import torch, transformers; print(torch.cuda.is_available(), transformers.__version__)"
```

第一次运行 DINO backbone 可能需要访问 Hugging Face。若 GPU 机器离线，
应提前配置模型缓存并记录 `HF_HOME`：

```bash
export HF_HOME=/path/to/huggingface-cache
```

## 4. 先准备四个 7-D PPO checkpoint

当前本地仓库没有可用于正式采集的四任务 7-D PPO checkpoint。每个任务都要
使用与采集阶段完全一致的 `pd_ee_delta_pose` 训练。

```bash
cd "$MANISKILL_ROOT"
source .venv/bin/activate

python rl/1_ppo_fast.py \
  --env-id PushCube-v1 \
  --control-mode pd_ee_delta_pose \
  --exp-name ppo_rm_pushcube \
  --num-envs 128 \
  --total-timesteps 20000000

python rl/1_ppo_fast.py \
  --env-id PokeCube-v1 \
  --control-mode pd_ee_delta_pose \
  --exp-name ppo_rm_pokecube \
  --num-envs 128 \
  --total-timesteps 20000000

python rl/1_ppo_fast.py \
  --env-id PlaceSphere-v1 \
  --control-mode pd_ee_delta_pose \
  --exp-name ppo_rm_placesphere \
  --num-envs 128 \
  --total-timesteps 20000000

python rl/1_ppo_fast.py \
  --env-id StackCube-v1 \
  --control-mode pd_ee_delta_pose \
  --exp-name ppo_rm_stackcube \
  --num-envs 128 \
  --total-timesteps 20000000
```

预期 checkpoint：

```text
runs/ppo_rm_pushcube/final_ckpt.pt
runs/ppo_rm_pokecube/final_ckpt.pt
runs/ppo_rm_placesphere/final_ckpt.pt
runs/ppo_rm_stackcube/final_ckpt.pt
```

不要仅凭 training return 选择 checkpoint。先确认 deterministic policy 能产生
success；否则正式采集会长期达不到 20 success。

## 5. 采集每任务 20 success + 20 fail

采集器：

```text
maniskill-ws/data_collect/collect_rm_episodes.py
```

它使用单环境，直接写出 Rsync raw transition pickle，不需要额外格式转换。
先从 `action_noise_std=0.05`、`random_action_prob=0.10` 开始：

```bash
cd "$MANISKILL_ROOT"
source .venv/bin/activate

python data_collect/collect_rm_episodes.py \
  --env-id PushCube-v1 \
  --checkpoint runs/ppo_rm_pushcube/final_ckpt.pt \
  --output-root "$REWARD_MODEL_ROOT/data" \
  --num-success 20 --num-fail 20 \
  --max-attempts 1000 \
  --action-noise-std 0.05 \
  --random-action-prob 0.10

python data_collect/collect_rm_episodes.py \
  --env-id PokeCube-v1 \
  --checkpoint runs/ppo_rm_pokecube/final_ckpt.pt \
  --output-root "$REWARD_MODEL_ROOT/data" \
  --num-success 20 --num-fail 20 \
  --max-attempts 1000 \
  --action-noise-std 0.05 \
  --random-action-prob 0.10

python data_collect/collect_rm_episodes.py \
  --env-id PlaceSphere-v1 \
  --checkpoint runs/ppo_rm_placesphere/final_ckpt.pt \
  --output-root "$REWARD_MODEL_ROOT/data" \
  --num-success 20 --num-fail 20 \
  --max-attempts 1000 \
  --action-noise-std 0.05 \
  --random-action-prob 0.10

python data_collect/collect_rm_episodes.py \
  --env-id StackCube-v1 \
  --checkpoint runs/ppo_rm_stackcube/final_ckpt.pt \
  --output-root "$REWARD_MODEL_ROOT/data" \
  --num-success 20 --num-fail 20 \
  --max-attempts 1000 \
  --action-noise-std 0.05 \
  --random-action-prob 0.10
```

调参规则：

- success 不够：降低 `random_action_prob` 和 `action_noise_std`，或换更好的 PPO。
- fail 不够：提高二者，例如 `0.15` 和 `0.10`。
- 每次重跑会重建该任务的 merged raw pickle；不要混合不同 checkpoint、
  control mode 或不同相机配置的数据。
- `collection_meta.json` 必须与实验记录一起保存。

## 6. 数据硬校验

进入 Rsync 环境：

```bash
cd "$REWARD_MODEL_ROOT"
source .venv/bin/activate

python scripts/validate_maniskill_rm_data.py \
  --tasks all \
  --minimum-episodes 20 \
  --output eval_results/maniskill_rm_data_validation.json
```

预期：

```text
pushcube     success= 20 fail= 20 errors=0
pokecube     success= 20 fail= 20 errors=0
placesphere  success= 20 fail= 20 errors=0
stackcube    success= 20 fail= 20 errors=0
```

只要有一个 `errors>0` 就停止，不要进入标注和训练。

## 7. 自动标注、RM 训练和统一评估

推荐一键运行：

```bash
cd "$REWARD_MODEL_ROOT"
source .venv/bin/activate

EPOCHS=100 \
NUM_WORKERS=16 \
NPROC_PER_NODE=1 \
PYTHON_BIN="$REWARD_MODEL_ROOT/.venv/bin/python" \
./scripts/run_maniskill_rm_pipeline.sh 2>&1 | tee rm_pipeline.log
```

该脚本依次执行：

1. 四任务 raw data 校验；
2. G-HMM 自动阶段发现和 dense potential 标注；
3. 四任务 RM 顺序训练；
4. held-out episode 统一质量评估。

多 GPU 单任务 DDP：

```bash
NPROC_PER_NODE=2 ./scripts/run_maniskill_rm_pipeline.sh
```

注意：当前脚本仍按任务顺序训练，不会同时占用四组 GPU。先用单 GPU 跑通一个
任务，再扩大并行度。

### 7.1 分阶段运行

需要定位问题时，不要使用一键脚本：

```bash
# 标注
python -m label.label \
  --tasks pushcube pokecube placesphere stackcube \
  --method auto

# 单任务训练
python train.py \
  --task_name pushcube \
  --prefix auto \
  --epochs 100 \
  --num_workers 16 \
  --use_gradient_checkpointing

# 四任务评估
python scripts/evaluate_maniskill_rm_quality.py \
  --tasks all \
  --checkpoint-template 'checkpoints/auto_{task}/best.pt' \
  --prefix auto \
  --device cuda \
  --split-ratio 0.9 \
  --seed 42 \
  --gamma 0.8 \
  --output eval_results/maniskill_rm_quality.json
```

## 8. 标签和 RM 质量如何判断

### 8.1 自动标签

每任务查看：

```text
data/<task>/auto_processed/<task>_dashboard.png
```

至少记录：

- PRA：success terminal 是否高于 fail terminal；
- Gap：success/fail terminal 分离；
- success monotonicity：成功轨迹势函数是否总体随进度上升；
- 是否出现所有轨迹几乎同一条曲线、阶段塌缩或大量零奖励。

自动标签由 G-HMM 在完整 20+20 数据上生成；它不是独立人工 ground truth。
因此标签自身的高 PRA 不能单独证明泛化。

### 8.2 RM held-out 指标

结果位于：

```text
eval_results/maniskill_rm_quality.json
```

重点字段：

- `supervision_fidelity`
  - `mse` / `mae`
  - `spearman_phi_vs_label`
  - `pairwise_rank_accuracy`
- `environment_alignment`
  - `spearman_phi_vs_dense`
  - `spearman_delta_phi_vs_delta_dense`
  - `progress_sign_agreement`
  - `spearman_pbrs_return_vs_dense_return`
- `trajectory_discrimination`
  - `terminal_pra`
  - `success_fail_gap`
  - `strict_terminal_gap`
- `temporal_quality`
  - `success_monotonicity_mean`
- `uncertainty`
  - ensemble uncertainty 与 label error 的相关性

脚本内置的诊断门槛：

- label Spearman ≥ 0.7；
- terminal PRA ≥ 0.9；
- progress sign agreement ≥ 0.55。

`all_pass=true` 只表示离线信号达到最低诊断要求，不代表论文结果已经复现。

### 8.3 PBRS 口径

评估使用与当前 SAC 一致的：

```text
F_t = gamma * phi(s_t) * (1 - done_t) - phi(s_{t-1})
gamma = 0.8
```

数据保存的是 post-transition observation，因此第一条 transition 缺少
`phi(s_0)`，离线 PBRS return 会跳过第一项；在线 SAC 不存在这个缺口。

## 9. 最终 policy utility（RM 真正验收）

离线质量评估后，仍需针对每个任务分别跑：

1. sparse reward baseline；
2. privileged ManiSkill dense reward baseline；
3. RM absolute reward；
4. RM PBRS reward；
5. fixed tau / adaptive tau；
6. 至少 3 个随机种子。

统一报告：

- `success_once`；
- success-AUC；
- 最终 success；
- env return；
- 固定 interaction budget；
- 每种配置的 seed 均值和标准差。

如果 RM 离线指标很好但 RM-only policy 很差，不能把 RM 判定为“质量好”；应继续
检查 reward scale、PBRS、归一化、distribution shift 和 RL 稳定性。

## 10. 故障排查

### CUDA 不可用

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
nvidia-smi
```

PyTorch CUDA wheel、驱动和服务器 CUDA runtime 必须兼容。

### `PlaceSphere-v1 not found`

这是 ManiSkill 任务注册/版本问题，不是数据路径问题。验证安装的是 ManiSkill3，
并在同一 Python 环境中先 `import mani_skill.envs`。

### `PinocchioModel` / `NoneType is not callable`

`pd_ee_delta_pose` 依赖机器人运动学支持。正式实验使用 Linux GPU 环境；
不要退回 4-D `pd_ee_delta_pos`，否则 action schema 与论文管线不一致。

### 只收集到 success

增加 `--random-action-prob` 或 `--action-noise-std`。不要将成功 episode 人工标成
fail。

### 只收集到 fail

降低噪声或继续 PPO 训练。随机策略不能提供足够的 success demo。

### DINO 下载失败

在联网机器预下载模型并复制 Hugging Face cache；记录 `HF_HOME` 和实际模型
revision。

### GPU OOM

先保持 `NPROC_PER_NODE=1`，降低训练 batch 配置或启用
`--use_gradient_checkpointing`。不要通过删除相机或缩短 state schema 临时绕过，
否则四任务不再可比。

## 11. 每次运行必须填写的实验记录

复制下面模板到 `eval_results/run_<date>_<host>.md`：

```markdown
# ManiSkill RM run

- Date:
- Host:
- GPU / driver:
- ManiSkill commit:
- Rsync commit:
- Python:
- PyTorch / CUDA:
- ManiSkill version:
- HF model revision/cache:
- Seed: 42
- Split ratio: 0.9
- Gamma: 0.8

## Data

| Task | PPO checkpoint | Success | Fail | Noise std | Random prob | Validation |
|---|---|---:|---:|---:|---:|---|
| pushcube | | 20 | 20 | | | |
| pokecube | | 20 | 20 | | | |
| placesphere | | 20 | 20 | | | |
| stackcube | | 20 | 20 | | | |

## Auto-label

| Task | Stages | PRA | Gap | Monotonicity | Notes |
|---|---:|---:|---:|---:|---|
| pushcube | | | | | |
| pokecube | | | | | |
| placesphere | | | | | |
| stackcube | | | | | |

## RM held-out quality

| Task | Best checkpoint | Label Spearman | Env Spearman | Terminal PRA | Strict gap | Progress sign | Pass |
|---|---|---:|---:|---:|---:|---:|---|
| pushcube | | | | | | | |
| pokecube | | | | | | | |
| placesphere | | | | | | | |
| stackcube | | | | | | | |

## Policy utility

| Task | Reward | Tau | Seeds | Success-AUC | Final success | Env return |
|---|---|---|---|---:|---:|---:|
| | sparse/dense/RM/PBRS | fixed/adaptive | | | | |

## Failures, retries, deviations

- 
```

## 12. 完成检查表

- [ ] 四任务 7-D PPO checkpoint 已记录。
- [ ] 每任务采集 20 success + 20 fail。
- [ ] `validate_maniskill_rm_data.py` 四任务均 `errors=0`。
- [ ] 四任务自动标签 dashboard 已人工检查。
- [ ] 四个 `checkpoints/auto_<task>/best.pt` 存在。
- [ ] `maniskill_rm_quality.json` 无 failed task。
- [ ] 环境、commit、seed、split、gamma 已记录。
- [ ] 后续 policy utility 多种子矩阵已计划或完成。
