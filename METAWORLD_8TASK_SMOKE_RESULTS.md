# MetaWorld 八任务端到端 Smoke 结果

运行日期：2026-07-27  
运行平台：macOS CPU（JAX CPU；MuJoCo offscreen rendering）  
MetaWorld：3.1.1  
Gymnasium：1.3.0  
MuJoCo：3.3.0  
JAX：0.4.35

代码基线：

- Rsync HEAD：`1531ad5f70e42d9334c34fbd27fb8900cc7f8dd0`
- SERL HEAD：`2613fcbc41f736847b6fa293ac01c2d6506e1339`
- 本报告对应两个仓库中的未提交 MetaWorld 集成改动。

## 1. 结论

八个目标任务的以下路径已经实际运行：

1. MetaWorld reset/step；
2. 官方 scripted expert 成功 rollout；
3. 每任务 20 success + 20 random fail、`corner2` RGB + 39-D state 数据采集；
4. Rsync prototype-stage 自动 dense potential 构造；
5. auto/dense/sparse 三种 reward 导出为 SERL replay；
6. 8 tasks × 3 rewards × fixed/adaptive tau = 48 个离线 SERL smoke run；
7. 8 tasks × dense/sparse × fixed/adaptive tau = 32 个在线环境 SERL/RLPD
   smoke run；
8. 每个在线 run 均生成 `run_config.json`、`metrics.csv` 和 agent checkpoint。

这里的“跑通”表示数据、reward、buffer、agent update、环境交互、evaluation、
checkpoint 路径均无异常，不表示 30 environment steps 已经学会任务。

## 2. 环境和数据验证

scripted expert 的单回合验证：

| Task | Success | Steps |
|---|---:|---:|
| Button-Press | 1 | 57 |
| Window-Open | 1 | 90 |
| Reach-Wall | 1 | 51 |
| Plate-Slide | 1 | 50 |
| Push | 1 | 59 |
| Coffee-Push | 1 | 56 |
| Stick-Push | 1 | 68 |
| Pick-Place | 1 | 58 |

正式数据硬校验结果：

| Task | Success episodes | Fail episodes | Total transitions | Errors |
|---|---:|---:|---:|---:|
| Button-Press | 20 | 20 | 5,184 | 0 |
| Window-Open | 20 | 20 | 5,733 | 0 |
| Reach-Wall | 20 | 20 | 4,917 | 0 |
| Plate-Slide | 20 | 20 | 5,007 | 0 |
| Push | 20 | 20 | 5,203 | 0 |
| Coffee-Push | 20 | 20 | 5,109 | 0 |
| Stick-Push | 20 | 20 | 5,486 | 0 |
| Pick-Place | 20 | 20 | 5,026 | 0 |

机器可读报告：

```text
eval_results/metaworld_data_validation.json
```

## 3. 自动 reward 构造质量

以下指标来自 G-HMM/prototype-stage pseudo label 本身，不是独立 ground truth：

| Task | Terminal PRA | Mean gap | Strict gap | Raw monotonicity |
|---|---:|---:|---:|---:|
| Button-Press | 1.000 | 2.661 | 2.290 | 0.987 |
| Window-Open | 1.000 | 4.351 | 0.295 | 0.980 |
| Reach-Wall | 0.963 | 0.963 | **-0.999** | 0.947 |
| Plate-Slide | 1.000 | 3.901 | 1.340 | 0.988 |
| Push | 1.000 | 3.895 | 2.299 | 0.966 |
| Coffee-Push | 1.000 | 3.659 | 1.311 | 0.975 |
| Stick-Push | 1.000 | 3.711 | 0.426 | 0.957 |
| Pick-Place | 1.000 | 4.164 | 2.175 | 0.884 |

Reach-Wall 的 PRA 虽然为 96.3%，但 strict gap 为负，表示至少存在一条失败
轨迹的 terminal potential 高于某条成功轨迹。正式 RM/策略实验中应将其标记为
reward-construction 风险，而不是按“100% 分离”报告。

机器可读报告：

```text
eval_results/metaworld_auto_label_quality.json
```

## 4. SERL smoke 矩阵

### 4.1 离线

- 配置数：48/48 完成；
- 每配置：2 次 agent update；
- reward：auto / MetaWorld dense / sparse success；
- tau：fixed / adaptive；
- adaptive 配置数：24；
- adaptive tau 参数：
  - critic loss threshold `0.05`
  - tolerance `0.2`
  - factor `1.1`
  - range `[0.001, 0.05]`
- 两次 update 后 adaptive tau：`0.004545`。

证据目录：

```text
serl/auto_research/logs/metaworld_serl/
```

### 4.2 在线环境交互

- 配置数：32/32 完成；
- 每配置：30 environment steps、15 次 agent update；
- reward：MetaWorld dense / sparse success；
- tau：fixed / adaptive；
- 每配置均加载对应 40-episode demo replay；
- 每配置均执行环境 evaluation 并保存 checkpoint；
- adaptive 配置数：16；
- adaptive tau 在 15 次 update 后为约 `0.001197`，证明动态调整路径实际执行。

证据目录：

```text
serl/auto_research/logs/metaworld_online_matrix_smoke/
```

30-step smoke 的 eval success 为 0 是预期结果：其目标是验证在线交互和更新，
不是训练收敛。

## 5. 尚未完成

1. 当前机器没有 CUDA Rsync RM 环境，因此尚未训练八个 DINO visual RM
   checkpoint。
2. 尚未运行八任务 RM/RM-PBRS 在线 reward；它们依赖上述 checkpoint。
3. 尚未执行图片同规模的 5M/10M/20M environment steps。
4. 尚未执行至少 3 seeds 的均值和置信区间。
5. 当前 SERL policy 使用 39-D MetaWorld state；RGB 只用于 Rsync RM reward，
   并非像素 DrQ policy。

GPU 阶段必须继续执行：

```bash
cd /path/to/reward-model
./scripts/run_metaworld_rm_pipeline.sh

cd /path/to/serl
export RSYNC_ROOT=/path/to/reward-model
MAX_STEPS=1000000 \
SEEDS="0 1 2" \
REWARD_MODES="dense sparse rm-pbrs" \
bash auto_research/scripts/run_metaworld_online_matrix.sh
```

完整操作说明见 `METAWORLD_8TASK_RUNBOOK.md`。
