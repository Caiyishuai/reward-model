# MetaWorld 八任务：数据与 Reward Model 运行手册

本文对应图中的八个 MetaWorld 任务：

| 图中名称 | MetaWorld 3.1 环境 | Reward-model task name |
|---|---|---|
| Button-Press | `button-press-v3-goal-observable` | `mw_button_press` |
| Window-Open | `window-open-v3-goal-observable` | `mw_window_open` |
| Reach-Wall | `reach-wall-v3-goal-observable` | `mw_reach_wall` |
| Plate-Slide | `plate-slide-v3-goal-observable` | `mw_plate_slide` |
| Push | `push-v3-goal-observable` | `mw_push` |
| Coffee-Push | `coffee-push-v3-goal-observable` | `mw_coffee_push` |
| Stick-Push | `stick-push-v3-goal-observable` | `mw_stick_push` |
| Pick-Place | `pick-place-v3-goal-observable` | `mw_pick_place` |

当前集成固定使用：

- MetaWorld `3.1.1`
- Gymnasium `1.3.0`
- MuJoCo `3.3.0`
- goal-observable 39-D state
- 4-D action
- `corner2` RGB camera
- MetaWorld reward function `v2`
- 20 success + 20 fail / task

## 1. 已验证范围

本机 CPU 已完成：

1. 八个环境 reset/step；
2. 八个 scripted expert 均达到 `info["success"] == 1`；
3. 八任务各 20 success + 20 fail RGB/state/wrist-wrench 采集并通过硬校验；
4. 输出 reward-model raw pickle 和 45-D state+wrench SERL stacked replay；
5. 八任务 auto/dense/sparse × fixed/adaptive tau，共 48 个离线 SERL smoke run；
6. 八任务 dense/sparse × fixed/adaptive tau，共 32 个在线 RLPD
   30-environment-step smoke run，均产生 checkpoint、`metrics.csv` 和运行配置。
7. 八任务 sparse × fixed/adaptive tau 的 learned force-gate 路径，共 16 个
   30-environment-step smoke run，均完成 15 次更新并保存 checkpoint。

尚需 Linux/NVIDIA GPU 完成：

1. 八任务正式 RM checkpoint；
2. 八任务 RM held-out 质量报告；
3. 长周期在线 SERL/RLPD 曲线；
4. 多随机种子统计。

这意味着“代码路径跑通”不等于已经复现图片中的最终曲线。

## 2. 环境安装

Reward-model 环境：

```bash
cd /path/to/reward-model
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[metaworld]"
```

检查：

```bash
python - <<'PY'
import gymnasium, metaworld, mujoco
print("gymnasium", gymnasium.__version__)
print("mujoco", mujoco.__version__)
print("tasks", len(metaworld.ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE))
PY
```

## 3. 数据采集

入口：

```text
scripts/collect_metaworld_rm_data.py
```

一次采集八任务正式数据：

```bash
cd /path/to/reward-model
source .venv/bin/activate

python scripts/collect_metaworld_rm_data.py \
  --tasks all \
  --output-root "$PWD" \
  --num-success 20 \
  --num-fail 20 \
  --max-episode-steps 200 \
  --image-size 128 \
  --failure-policy random \
  --reward-function-version v2 \
  --wrench-filter-alpha 0.2 \
  --wrench-force-clip 100 \
  --wrench-torque-clip 10 \
  --seed 42
```

success 使用 MetaWorld 官方 scripted policy；fail 使用 random policy，并且只接收
整条 episode 从未成功的轨迹。

每任务输出：

```text
reward-model/data/mw_<task>/
├── success_raw.pkl
├── fail_raw.pkl
├── serl_dense.pkl
├── serl_sparse.pkl
└── collection_meta.json
```

Reward-model raw transition：

```text
observations.state             (39,) float32，执行 action 后的状态
observations.corner2           (128,128,3) uint8
observations.wrist_wrench       (6,) float32，[Fx,Fy,Fz,Tx,Ty,Tz]
previous_observations.state    (39,) float32
previous_observations.wrist_wrench (6,) float32
actions                        (4,) float32
env_rewards                    MetaWorld v2 dense reward
sparse_rewards                 info["success"] 的 0/1
contact_force                  (N_contact,3)，逐接触点世界系力
max_contact_force              当前仿真步单接触点最大力范数
dones                          episode 最后一帧为 1
infos.succeed                  仅成功 episode 最后一帧为 True
```

`wrist_wrench` 不是 MetaWorld XML 中已有的 sensor（默认 `nsensor=0`）。采集器从
MuJoCo 已求解 contact constraint 中提取每个外部接触力，将力矩平移到
`endEffector` site，求和后旋转到腕部坐标系，并使用 EMA (`alpha=0.2`) 稳定信号。
机器人端使用 `right_l6` 及其手和夹爪子树；内部机器人接触不会计入。

`serl_dense.pkl` 和 `serl_sparse.pkl` 中的 policy observation 为：

```text
[MetaWorld goal-observable state (39), wrist_wrench (6)] = 45-D
```

## 4. 数据校验

```bash
cd /path/to/reward-model
source .venv/bin/activate

python scripts/validate_metaworld_rm_data.py \
  --tasks all \
  --minimum-episodes 20 \
  --output eval_results/metaworld_data_validation.json
```

预期所有任务：

```text
success=20 fail=20 errors=0
```

任何 error 都必须先修复，不能继续训练。

## 5. 自动 reward 构造与 RM 训练

八任务已经注册在 `data/common.py`。完整 GPU 流水线：

```bash
cd /path/to/reward-model
source .venv/bin/activate

EPOCHS=100 \
NUM_WORKERS=16 \
NPROC_PER_NODE=1 \
PYTHON_BIN="$PWD/.venv/bin/python" \
./scripts/run_metaworld_rm_pipeline.sh 2>&1 | tee metaworld_rm_pipeline.log
```

脚本依次执行：

1. 20+20 数据硬校验；
2. G-HMM 阶段发现；
3. success/fail dense potential 自动标注；
4. 导出 auto/dense/sparse 三套 SERL replay并保存 auto-label 质量报告；
5. DINO + temporal RM 训练；
6. held-out episode RM 质量评估。

每任务输出：

```text
data/mw_<task>/auto_processed/
checkpoints/auto_mw_<task>/best.pt
data/mw_<task>/serl_auto.pkl
data/mw_<task>/serl_dense.pkl
data/mw_<task>/serl_sparse.pkl
```

统一质量报告：

```text
eval_results/metaworld_auto_label_quality.json
eval_results/metaworld_rm_quality.json
```

核心指标：

- RM vs auto-label MSE、Spearman、pairwise rank accuracy；
- RM potential vs MetaWorld dense reward；
- RM progress direction vs dense-reward progress；
- success/fail terminal PRA 和 strict gap；
- success monotonicity；
- ensemble uncertainty。

## 6. 离线 SERL 全矩阵

该矩阵验证采集数据可以进入 SERL SACAgent，并覆盖：

- 8 tasks；
- auto / dense / sparse reward；
- fixed / adaptive tau。

共 48 个配置/seed：

```bash
cd /path/to/serl
export RSYNC_ROOT=/path/to/reward-model

MAX_UPDATES=3000 \
TIME_LIMIT_MIN=30 \
BATCH_SIZE=256 \
UTD_RATIO=4 \
SEEDS="0 1 2" \
bash auto_research/scripts/run_metaworld_serl_matrix.sh
```

`RSYNC_ROOT` 是外部 SERL 脚本保留的兼容变量名；它现在应指向本
`reward-model` 仓库。

结果：

```text
serl/auto_research/logs/metaworld_serl/
├── mw_<task>__<reward>__<tau>__seed<seed>.csv
└── mw_<task>__<reward>__<tau>__seed<seed>.log
```

这里是 demo-only 离线训练，用于检查 reward/tau/数据路径；不能用它生成图片中的
environment-step success 曲线。

## 7. 在线 SERL/RLPD

单任务：

```bash
cd /path/to/serl

auto_research/venv_serl/bin/python \
  auto_research/scripts/train_serl_metaworld.py \
  --task button-press \
  --reward-mode dense \
  --demo-path /path/to/reward-model/data/mw_button_press/serl_dense.pkl \
  --adaptive-tau \
  --max-steps 1000000 \
  --seed 0 \
  --output-dir auto_research/logs/metaworld_online/button_dense_adaptive_seed0
```

支持的在线 reward：

```text
dense      MetaWorld v2 privileged dense reward
sparse     info["success"] 0/1
rm         Rsync RM absolute potential + goal bonus
rm-pbrs    gamma*phi(s')*(1-done)-phi(s) + goal bonus
```

RM PBRS 示例：

```bash
auto_research/venv_serl/bin/python \
  auto_research/scripts/train_serl_metaworld.py \
  --task button-press \
  --reward-mode rm-pbrs \
  --demo-path /path/to/reward-model/data/mw_button_press/serl_auto.pkl \
  --rm-checkpoint /path/to/reward-model/checkpoints/auto_mw_button_press/best.pt \
  --rsync-root /path/to/reward-model \
  --rm-device cuda \
  --rm-gamma 0.99 \
  --rm-goal-bonus 1.0 \
  --adaptive-tau \
  --max-steps 1000000 \
  --seed 0 \
  --output-dir auto_research/logs/metaworld_online/button_rm_pbrs_adaptive_seed0
```

在线 dense/sparse/RM-PBRS × fixed/adaptive × 三种子矩阵：

```bash
export RSYNC_ROOT=/path/to/reward-model
MAX_STEPS=1000000 \
SEEDS="0 1 2" \
REWARD_MODES="dense sparse rm-pbrs" \
bash auto_research/scripts/run_metaworld_online_matrix.sh
```

矩阵脚本显式使用论文 adaptive-tau 参数，不依赖 `SACAgent` 的旧默认值：

```text
critic_loss_threshold = 0.05
tau_adjust_tolerance  = 0.2
tau_adjust_factor     = 1.1
tau range             = [0.001, 0.05]
```

先执行八任务短 smoke：

```bash
export RSYNC_ROOT=/path/to/reward-model
MAX_STEPS=30 \
MAX_EPISODE_STEPS=50 \
RANDOM_STEPS=5 \
TRAINING_STARTS=16 \
BATCH_SIZE=32 \
UTD_RATIO=1 \
EVAL_PERIOD=15 \
EVAL_EPISODES=1 \
SAVE_PERIOD=30 \
SEEDS="0" \
REWARD_MODES="dense sparse" \
OUTPUT_ROOT="$PWD/auto_research/logs/metaworld_online_matrix_smoke" \
bash auto_research/scripts/run_metaworld_online_matrix.sh
```

每个 run 输出：

```text
run_config.json
metrics.csv
checkpoints/agent_step_*.msgpack
```

`metrics.csv` 包含 environment step、episode success、eval success rate、dense
return、critic/actor loss 和当前 tau，可直接用于画与用户图片同口径的 success
rate vs environment steps 曲线。

## 8. 力觉门控与视觉输入

真机 `hil-serl` 的输入约定是 `tcp_force(3) + tcp_torque(3)`，由
`SERLObsWrapper` 与其他 proprioception 一起展平。MetaWorld 对应实现：

```text
scripts/train_serl_metaworld_force.py
scripts/run_metaworld_force_serl_matrix.sh
```

网络先按 `[30,30,30,3,3,3]` 缩放六维 wrench，再经 32-D force projection；
另一个 learned sigmoid gate 根据 `state+wrench` 对 force feature 逐维门控，
最后送入 actor/critic 各自的 MLP。原始 39-D state 不经过 gate，因此无接触或
力信号不可靠时不会丢失基础状态信息。

八任务 sparse、fixed/adaptive tau 正式矩阵：

```bash
cd /path/to/reward-model
export SERL_ROOT=/path/to/serl
export PYTHON_BIN="$SERL_ROOT/auto_research/venv_serl/bin/python"

MAX_STEPS=1000000 \
SEEDS="0 1 2" \
REWARD_MODES="sparse" \
TAU_MODES="fixed adaptive" \
bash scripts/run_metaworld_force_serl_matrix.sh
```

MetaWorld 本身不强制 visual policy；上述两个 trainer 仍是 legacy state-based
路径。新增的 visual DrQ 路径是独立实现：

```text
scripts/collect_metaworld_visual_demos.py
scripts/validate_metaworld_visual_demos.py
scripts/train_serl_metaworld_visual.py
scripts/run_metaworld_visual_drq_matrix.sh
```

policy / replay 的稳定 dict schema：

```text
state    (19,) float32
wrist_1  (128,128,3) uint8  <- behindGripper（hand body）
wrist_2  (128,128,3) uint8  <- gripperPOV（hand body）
front    (128,128,3) uint8  <- corner2
```

19-D robot-only state 精确定义：

```text
0:3    endEffector site 世界系 xyz（m）
3:6    endEffector 姿态 intrinsic XYZ Euler / roll-pitch-yaw（rad）
6:9    endEffector 世界系线速度 xyz（m/s）
9:12   endEffector 世界系角速度 xyz（rad/s）
12:18  endEffector 腕系 [Fx,Fy,Fz,Tx,Ty,Tz]（N, Nm）
18     rightEndEffector 与 leftEndEffector site 世界系距离（m）
```

速度由 `mj_objectVelocity(..., mjOBJ_SITE, endEffector, local=0)` 获取，并从
MuJoCo 的 angular-linear 顺序重排为 linear-angular。gripper state 只读取机器人
tip site；整个构造不读取 object qpos、目标位置或 39-D goal-observable state。
scripted policy 采集时仍接收官方 observation 以生成专家动作，但写入 demo 和在线
policy 的 observation 只有上述 robot-only dict。

采集与硬校验（每任务恰好 20 条成功 demo，不采失败轨迹）：

```bash
python scripts/collect_metaworld_visual_demos.py \
  --tasks all --output-root data --num-demos 20 \
  --image-size 128 --force-filter ema --wrench-filter-alpha 0.2

python scripts/validate_metaworld_visual_demos.py \
  --tasks all --data-root data --expected-episodes 20
```

每任务输出 `data/mw_<task>/visual_drq/success_demos.pkl.gz` 与
`metadata.json`。pickle 是 stacked dict arrays，包含 observations、
next_observations、actions、sparse rewards、masks、dones、episode index/step/
seed。validator 检查 dtype/shape、终止边界、仅 terminal sparse reward、seed 与
连续 transition 的三路图像/state 一致性。

训练默认 sparse + `ema` + `learned_gate` + `resnet-pretrained`。force 可选
`ema|none`，融合可选 `learned_gate|concat|none`；learned gate 位于 actor 和
critic 共用的 DrQ encoder，参数路径和训练前后最大变化会写入 config/summary。
MemoryEfficientReplayBuffer 运行时增加长度为 1 的 frame-stack 维度，磁盘 schema
仍保持 HWC。

```bash
export SERL_ROOT=/path/to/serl
MAX_STEPS=1000000 \
SEEDS="0 1 2" \
TAU_MODES="fixed adaptive" \
EVAL_PERIOD=10000 \
EVAL_EPISODES=10 \
bash scripts/run_metaworld_visual_drq_matrix.sh
```

CPU smoke 使用 `ENCODER_TYPE=small`；正式 GPU 推荐默认
`resnet-pretrained`。如果随机目标在三路图像中不可见，策略面对的是 POMDP；
本实现不会通过泄漏 goal/object state 来规避该问题。

visual trainer 的周期评估使用独立 MetaWorld env、相同三相机 robot-only wrapper
和相同 wrench filter，以 deterministic (`argmax=True`) policy action rollout；
评估 transition 不写入 online/demo replay。`metrics.csv` 的
`train_success` 只在训练 episode 完成时填写，`eval_success_rate` 只在评估周期
填写，避免 reset 后的 episode 状态覆盖真实结果。`--eval-period 0` 可显式关闭
评估。

## 9. 公平对比要求

所有方法必须固定：

- 相同任务 seed；
- 相同 20+20 demo；
- 相同总 environment steps；
- 相同 eval period 和 eval episodes；
- 相同 SAC/RLPD 网络、batch、UTD；
- 只有 reward 或 tau 配置变化；
- 至少 3 seeds，曲线报告 mean ± std。

评估始终使用 MetaWorld `info["success"]`，不能用 RM reward 判断成功。

## 10. 运行记录模板

```markdown
# MetaWorld 8-task run

- Date:
- Host / GPU:
- MetaWorld commit/version:
- SERL commit:
- Rsync commit:
- Seeds:
- Environment steps:
- Eval episodes:

| Task | Data validation | Auto-label PRA | RM checkpoint | RM terminal PRA |
|---|---|---:|---|---:|
| Button-Press | | | | |
| Window-Open | | | | |
| Reach-Wall | | | | |
| Plate-Slide | | | | |
| Push | | | | |
| Coffee-Push | | | | |
| Stick-Push | | | | |
| Pick-Place | | | | |

| Task | Reward | Tau | Seed | Best success | Final success | Success-AUC |
|---|---|---|---:|---:|---:|---:|
| | dense/sparse/rm-pbrs | fixed/adaptive | | | | |
```

## 11. 重要限制

- 本地 smoke 证明代码路径能运行，不证明收敛到图片结果。
- MetaWorld 官方 scripted policy 只用于 demo，不能用于在线 eval。
- auto label 是 pseudo ground truth；RM 高拟合分数不代表 RL utility 高。
- RM 与 JAX 同时使用 GPU 时已关闭 JAX 全显存预分配，但仍需监控显存。
- 图片中的不同颜色没有图例，不能仅凭图片判断具体算法对应关系。
