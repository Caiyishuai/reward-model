# Rsync 自动奖励标注

`label/` 是 Rsync 的奖励监督信号构建子系统：输入只有 episode 级成功/失败标签的机器人演示，自动生成逐步稠密奖励，再导出为 Rsync 奖励模型训练所需的 LeRobot 风格 pickle。

> 范围说明：本目录负责“生成和评估奖励标签”，不包含视觉奖励模型本体的训练与在线推理。奖励模型训练入口是仓库根目录的 `train.py`，ManiSkill 在线强化学习集成位于 `sim/`。

## 在完整系统中的位置

```text
maniskill-ws / SERL / 真机采集
        │ raw demo pickle
        ▼
reward-model/label
  ├─ HMM 自动标注（当前生产入口）
  ├─ 手工势函数标注（对照）
  └─ 候选策略搜索与泛化评估
        │ data/<task>/{auto,manual}_processed/*.pkl
        ▼
reward-model/train.py
        │ reward-model checkpoint
        ▼
reward-model/sim 或外部 RL 训练器
```

当前接口主要通过文件连接，各仓库之间没有 Python 包级依赖。

## 快速开始

在 Rsync 仓库根目录执行：

```bash
cd /path/to/reward-model
uv sync

# 自动标注一个或多个任务
uv run python -m label.label --tasks pushcube --method auto
uv run python -m label.label --tasks button usb --method auto

# 标注任务注册表中的全部任务
uv run python -m label.label --tasks all --method auto

# 使用人工目标点定义的势函数方法
uv run python -m label.label --tasks button --method manual
```

任务名、原始数据路径、相机名和状态切片统一配置在 `data/common.py::TASK_REGISTRY`。数据根目录默认是 `<Rsync>/data`，也可通过环境变量覆盖：

```bash
export RM_DATA_DIR=/absolute/path/to/data
```

自动标注产物位于：

```text
data/<task>/auto_processed/
├── success_lerobot.pkl
├── fail_lerobot.pkl
└── <task>_dashboard.png
```

随后可训练奖励模型：

```bash
uv run python train.py \
  --task_name pushcube \
  --prefix auto \
  --epochs 100 \
  --num_workers 12
```

## 输入数据

输入可以是扁平的 `list[dict]`，也可以是由多个 transition list 组成的列表。`data.common.get_episodes` 使用 `dones` 切分 episode。

每个 transition 至少需要：

```python
{
    "observations": {
        "state": np.ndarray,       # (D,) 或 (1, D)
        "<camera_key>": np.ndarray # H x W x 3 或 1 x H x W x 3
    },
    "actions": np.ndarray,
    "rewards": float,             # 会被标注结果覆盖
    "dones": bool | int,
    "infos": {"succeed": bool},   # 可选
}
```

真机数据默认使用 19 维状态：

```text
0       gripper
1:4     force
4:7     end-effector position
7:10    rotation
10:13   torque
13:16   linear velocity
16:19   angular velocity
```

`pushcube` 是仿真特例：使用 17 维状态、`wrist`/`third` 相机，不使用力信号。不要假设所有任务都共享同一状态布局，应以 `TASK_REGISTRY` 为准。

只应加载可信 pickle；Python pickle 反序列化可以执行任意代码。`data.common.load_pickle` 会限制读取路径必须位于数据根目录或当前工作目录内。

## 输出格式

`success_lerobot.pkl` 和 `fail_lerobot.pkl` 是列式字典：

```python
{
    "observation.state": np.ndarray,
    "observation.images.<camera_key>": np.ndarray,
    "action": np.ndarray,
    "next.reward": np.ndarray,
    "next.done": np.ndarray,
    "next.success": np.ndarray,
    "episode_index": np.ndarray,
    "frame_index": np.ndarray,
    "index": np.ndarray,
}
```

其中 `next.reward` 是本模块的核心产物，也是 `train.py` 的监督目标。

## 两条自动标注路线

### 当前生产入口：HMM 阶段发现

`python -m label.label --method auto` 实际调用 `auto_label.py`：

1. 从成功演示中抽取状态以及可选的力变化特征。
2. 训练只能自环或向前转移的 Gaussian HMM。
3. 未指定阶段数时，在 `[2, max_stages]` 中用 BIC 选择。
4. 从阶段切换点估计位置、力、夹爪和阶段时长分布。
5. 用 Mahalanobis 距离、接触一致性、夹爪约束和超时惩罚生成稠密奖励。
6. 计算 PRA、成功/失败间隔和单调性，生成 dashboard。
7. 导出 LeRobot 风格 pickle。

奖励的实现形式为：

```text
reward = stage_index
       + progress
       × force_match
       × gripper_match
       × duration_penalty
```

### 研究最优路线：GloballyConsistent

`strategies/globally_consistent.py` 实现了实验报告中的最终候选：

```text
P(success | step, episode)
  = 0.3 × step-level HGBC
  + 0.2 × KNN
  + 0.5 × trajectory-level HGBC

reward(t)
  = weighted_cumulative_mean(P) × (0.7 + 0.3 × t/T)
```

它综合 state、速度、加速度、action、action velocity、相对时间、累计位移和 trajectory summary。`report.md` 记录的 8 任务结果为 LOO-sAUC 0.985、Composite 0.9877。

但这条路线当前只接入策略评估，尚未接入 `label.label` 的导出流程。因此：

- `--method auto` 并不会使用 HGBC + KNN；
- `report.md` 的最佳数字不能直接代表当前默认 HMM 产物；
- 如要用该策略训练奖励模型，还需要补充 strategy → LeRobot export 的生产入口。

## 质量评估

```bash
# 默认评估 GloballyConsistentStrategy，包含 leave-one-out
uv run python -m label.metric

# 指定任务；跳过 LOO 可显著加速
uv run python -m label.metric --tasks button pushcube --no-loo

# 兼容旧入口
uv run python -m label.eval_labeling

# 对策略池进行 benchmark
uv run python -m scripts.benchmark_labeling_strategies \
  --task pushcube \
  --no-loo
```

主要指标：

- **PRA**：成功 episode 最终奖励高于失败 episode 的两两排序准确率。
- **sAUC**：时间对齐后，每个时间切片的成功/失败 AUC。
- **LOO-sAUC**：每条 episode 都由未见过该 episode 的模型标注，衡量真实泛化。
- **WinRank**：随机时间窗口内成功奖励和高于失败奖励和的比例。
- **Mono**：成功奖励曲线的近似单调比例。
- **Gap**：最差成功最终奖励减去最好失败最终奖励。
- **Composite**：`0.40×LOO + 0.30×WinRank + 0.15×PRA + 0.15×Mono`。

`report.md` 说明了为什么仅看训练集 PRA 会产生虚假的 100%，以及项目如何逐步引入统一归一化、多任务和 leave-one-out 评估。

## 文件与目录

```text
label/
├── README.md
│   本文档。
├── label.py
│   统一 CLI；按 --method 分发到 auto 或 manual 管线。
├── auto_label.py
│   HMM 阶段发现、阶段目标统计、稠密奖励、dashboard 和导出。
├── manual_label.py
│   调用 data/preprocess.py 中的人工目标点势函数标注，并导出数据。
├── metric.py
│   PRA、sAUC、LOO-sAUC、WinRank、单调性、coherence 和综合评分。
├── eval_labeling.py
│   metric.py 的向后兼容包装与默认八任务评估入口。
├── report.md
│   自动策略搜索、失败实验、指标演进和最终结果报告。
├── experiment_ledger.jsonl
│   每轮实验的假设、指标、保留/丢弃状态和原因。
└── strategies/
    ├── __init__.py
    │   对外导出策略和公共工具。
    ├── _base.py
    │   LabelingStrategy 接口、StrategyConfig、特征抽取、平滑和归一化。
    ├── registry.py
    │   构造用于自动搜索/benchmark 的候选策略池。
    ├── hmm.py
    │   HMM baseline 策略封装。
    ├── globally_consistent.py
    │   HGBC + KNN + trajectory prior 的当前研究最优策略。
    ├── contrastive.py
    │   ContrastiveDistance 与 TemporalContrastive 策略。
    ├── attribution.py
    │   ProgressEstimator 与 ReturnDecomposition 策略。
    ├── hybrid.py
    │   HMM/contrastive/progress 的混合与 ensemble 策略。
    └── advanced.py
        PotentialBased、Discriminative 和 OptimalTransport 策略。
```

本目录还依赖以下仓库级文件：

- `data/common.py`：任务注册、状态布局、episode 切分和 pickle I/O。
- `data/preprocess.py`：人工势函数奖励。
- `data/convert_data.py`：manual 路线的 LeRobot 转换。
- `data/dataset.py`：奖励模型训练数据集。
- `train.py`：奖励模型训练。
- `reward_model.py`：视觉时序奖励模型。
- `sim/`：ManiSkill 观测适配、奖励模型推理和 SAC。

## 与另外两个项目的接口

### `maniskill-ws` → `reward-model/label`

ManiSkill 原始 demo 通常是 HDF5/JSON 或 LeRobot parquet，而本模块读取 transition pickle。需要先完成：

```text
ManiSkill trajectory
  → 回放/渲染并整理 observation、action、done、success
  → success/fail raw pickle
  → 在 data/common.py 注册 task
  → label.label
```

四个论文任务现已在 `data/common.py` 注册。GPU 数据采集入口是
`maniskill-ws/data_collect/collect_rm_episodes.py`，固定输出 17-D
`eef+joint` state、7-D action、`hand_camera`/`base_camera`、episode 边界，
并将 ManiSkill dense reward 单独保存在 `env_rewards`。正式标注前运行：

```bash
python scripts/validate_maniskill_rm_data.py --tasks all
```

该检查要求每个任务至少 20 success + 20 fail，且拒绝旧采集器产生的跨环境
batch 图像、4-D action、缺失 dense reward 或错误 episode 边界。完整标注、
训练和 hold-out 评估可用 `scripts/run_maniskill_rm_pipeline.sh` 在 Linux/NVIDIA
环境一次执行。

### `serl` → `reward-model/label`

两者通过 raw transition pickle 间接兼容。SERL 采集数据至少要保留：

- `observations.state` 和所需相机；
- `actions`；
- `dones`；
- episode 成功/失败归类。

若 SERL demo 字段、状态维度或相机 key 不同，需要转换并更新 `TASK_REGISTRY`，不能仅复制文件后假设自动兼容。

### `reward-model/label` → SERL

标注后的 `next.reward` 首先用于训练 Rsync 奖励模型。训练出的 checkpoint 才是在线 SERL/SAC 使用的接口；`label/` 产物本身不是可直接调用的在线 reward function。

## 与论文的关系

论文 *Rsync: Reward Manifold-Aware Adaptive Synchronization for Sample-Efficient Real-World Reinforcement Learning* 的核心强化学习贡献是：根据奖励流形状态自适应调整 target-network 同步，而不是始终使用固定 Polyak 系数。本目录提供该方法所依赖的奖励监督构建与质量评估基础，但不实现 Rsync 的 adaptive synchronization 更新规则；该规则应在训练 agent 的代码中验证。

因此，单独运行本目录只能复现：

- 从二值演示构建逐步奖励标签；
- 自动/人工标注对照；
- 多种标注策略及其泛化指标。

它不能单独复现：

- 视觉奖励模型训练和 checkpoint；
- Rsync 自适应同步算法；
- ManiSkill 或真机策略训练；
- 论文表格中的端到端成功率和消融。

## 当前复现状态与已知缺口

- 仓库未提交论文全部原始 demo，初次克隆不能直接运行所有任务。
- 默认生产管线与实验最优策略尚未统一。
- `label.py` 顶部旧示例写的是 `--task`，实际参数是 `--tasks`；以本文命令为准。
- PushCube 的“20 success + 20 fail”论文约束需由数据准备阶段显式采样；现有 benchmark 记录使用过更多 episode。
- 纯奖励模型驱动的 ManiSkill SAC 尚不能仅凭本目录证明达到论文结果。
- LOO 会对每条 episode 重新拟合策略，完整八任务评估耗时较长。

论文与当前代码还存在以下需要在正式复现前统一的差异：

- 论文 Eq. (7) 写作 `exp(-0.5 × d_Mah²)`；`auto_label.py` 当前先取平方根距离，再计算 `exp(-0.5 × d_Mah)`。
- 论文写 DINOv3 且解冻最后两层；当前 `reward_model.py` 默认是
  `facebook/dinov2-small`。论文 checkpoint 与本仓库默认实现仍需单独核对。
- 论文在线奖励是 PBRS 差分 `γφ(s') - φ(s) + goal_bonus`；Rsync 当前仿真主线主要使用绝对 RM 输出与环境奖励加权，严格来说不是同一奖励公式。
- 论文默认 adaptive synchronization 参数为 `ξ=0.05`、`δτ=0.2`、`ρ=1.1`、`τ∈[0.001,0.05]`。不同 SAC/DrQ 入口的默认值并不全部一致。
- Rsync `GOAL.md` 已记录纯 RM reward 的 PushCube SAC 峰值 success 为 25%、末值 0%，尚未达到约 94% 的环境奖励 baseline。
- 当前代码和数据不足以严格重建论文中的 LIBERO、π0.5 residual、全部四个 ManiSkill 任务、九个真机任务及相应多种子统计。

要完整复现论文，仍需锁定 demo 数据版本、奖励模型 checkpoint/超参数、RL 配置、随机种子和最终实验统计脚本。

