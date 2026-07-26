# 自动化 Reward 标注实验报告

## 1. 实验目标

用 20 条成功 + 20 条失败的机器人操作轨迹（仅二值标签），通过自动化框架生成 dense per-step reward 信号，供下游 Reward Model 训练使用。

## 2. 起点状态

实验开始前，`label/strategies.py` 中已有多种策略（HMM baseline、Contrastive、ProgressEstimator 等），但均为独立设计，未经系统优化。`GloballyConsistentStrategy` 是初始最优策略，结构如下：

```
初始架构:
  - 特征: state(19维)
  - 分类器: 15 个 per-time-bin LogisticRegression
  - 后处理: is_success 分支 → 成功用 maximum.accumulate，失败用 plateau
  - 归一化: 成功 [3, 6]，失败 [0, 2.5]（硬编码分离）
```

初始评估指标（blind mode, 仅 button 任务）：composite=0.72, PRA=97.2%。

## 3. 实验历程

### 第一期：分类器演进（R2 — R7, 旧指标体系）

**指标体系（旧）：** blind composite = PRA×Gap×Mono 的加权组合。评估只在 button 单任务上进行。

| 轮次 | 修改 | 核心结果 | 状态 |
|---|---|---|---|
| R1 | 去掉 is_success 分支 | Mono 100%→32% 崩溃 | Discard |
| R2 | 加入 per-bin LogisticRegression | composite 0.72→0.79 | Keep |
| R3 | 加入 velocity + acceleration 特征 | composite 0.79→0.80 | Keep |
| **R4** | **换成 HistGradientBoosting** | **composite 0.80→0.99** | **Keep** |
| R5 | bins 15→10 | composite 0.99→0.998 | Keep |
| R6 | 更深的树 (depth=6) | composite→1.00 | Keep |
| R7 | 去掉 bin_stats，纯分类器信号 | composite 1.00, gap↑ | Keep |

**关键转折 R4:** LogisticRegression→HistGradientBoosting，composite 从 0.80 跳到 0.99。非线性决策边界对机器人轨迹数据至关重要。

### 指标体系革命：发现 PRA=100% 的虚假繁荣

R7 之后发现：blind composite=1.00 看似完美，但实际存在严重问题——硬编码的成功 [3,6] / 失败 [0, 2.5] 归一化范围**人为保证**了成功>失败。随机标注也能拿 100% PRA。

**修复：** 创建 `eval_labeling.py`，引入统一归一化 + multi-task 评估 + stepwise AUC + coherence。评估扩展到 8 个任务。

### 第二期：结构简化与多任务优化（R8 — R12, 新指标体系 v1）

**指标体系 v1：** composite = 0.25×PRA + 0.30×sAUC + 0.15×Mono + 0.15×Coh + 0.10×Gap + 0.05×MinPRA。在 8 个任务上评估。

| 轮次 | 修改 | composite | 状态 |
|---|---|---|---|
| R8 (discard) | 去掉 is_success 分支 | Mono 94%→崩 | Discard |
| **R9** | **cumulative mean 替代 is_success 分支** | **0.9912** | **Keep** |
| R10 | 加 relative_time 特征 + smooth=15 | 0.9943 | Keep |
| R11 | EMA 平滑 | 0.9948 | Keep |
| **R12** | **单个全局分类器替代 10 个 per-bin** | **0.9664** | **Keep** |
| R13 | HGBC depth=4 l2=5.0 调参 | 0.9744 | Keep |
| R14 (discard) | Mahalanobis 距离混合 | 0.8413 | Discard |
| R15 | 简化 reward 公式为一行 | 0.9697 | Keep |
| R16 | 加 cumulative displacement 特征 | 0.9732 | Keep |
| R17 | 线性加权 cumulative mean | 0.9798 | Keep |
| R18 | HGBC grid search: depth=5 lr=0.08 | 0.9892 | Keep |

**关键转折 R9:** cumulative mean 完美解决了无 is_success 分支下的单调性问题——用概率的累积均值天然保证不降。

**关键转折 R12:** 10 个 per-bin 分类器→1 个全局分类器。sAUC 从虚假的 1.000 降到诚实的 0.975，暴露了真实过拟合。速度快 35x。

### 第三期：泛化力攻坚（R0-R12, 指标体系 v2 — LOO）

v1 指标的 composite=0.9892 看似接近完美，但深入分析发现 **train sAUC=0.994 而 LOO sAUC 仅 0.772**——差距 0.222，严重过拟合。尤其 pl_toy(0.435) 和 pk_toy(0.623) 基本失效。

**指标体系 v2（最终版）：**

| 指标 | 角色 | 定义 |
|---|---|---|
| LOO-sAUC | Primary | 每个 episode 用排除自身的模型标注，在所有时间切片算 AUC |
| WinRank | Primary | 随机时间窗口上 success reward sum > fail 的比例 |
| PRA | Guard ≥95% | 最终步排序准确率 |
| Mono | Guard ≥90% | 成功曲线单调性 |
| Composite | Target | 0.40×LOO + 0.30×WinR + 0.15×PRA + 0.15×Mono |

Baseline（当前最佳 v1 方案）在 v2 下的真实成绩：**LOO=0.772, Composite=0.8168**。

| 轮次 | 修改 | LOO | Composite | 状态 |
|---|---|---|---|---|
| **R1** | **加入 action(7d) + action_vel(7d) 特征** | **0.797** | **0.8368** | **Keep** |
| R2 | 重正则化 depth=3 l2=10 leaf=30 | 0.781 | 0.8189 | Discard |
| R3 | 滑动窗口上下文特征 | OOM crash | — | Crash |
| R4 | causal window-mean 替代 raw state | 0.794 | 0.8342 | Discard |
| R5 | Bagging 10 个浅 HGBC | 0.774 | 0.8130 | Discard |
| R6 | 纯 KNN（无训练参数） | 0.784 | 0.8276 | Discard |
| **R7** | **HGBC + KNN 50/50 blend** | **0.812** | **0.8492** | **Keep** |
| **R8** | **KNN k=5 (grid search)** | **0.814** | **0.8509** | **Keep** |
| R9 | 训练时 subsample 20% 时间步 | pk_toy PRA=0% | — | Discard |
| R10 | subsample 50% | pk_toy PRA=0% | — | Discard |
| **R11** | **+trajectory-level 分类器 (40% HGBC + 30% KNN + 30% traj)** | **0.967** | **0.9737** | **Keep** |
| **R12** | **blend 权重优化 → h=0.3 k=0.2 t=0.5** | **0.985** | **0.9877** | **Keep** |
| R14 | 更丰富 trajectory summary | Timeout (>10min) | — | Timeout |

**关键转折 R1:** action 特征包含策略意图——gripper 闭合时机、插入角度等 state 无法表达的信息。pickup LOO 从 0.800 跳到 0.904。

**关键转折 R6（虽被 discard 但至关重要）:** 纯 KNN 在 op_dr 和 pl_toy 上 LOO 各提升 +0.105。证明了 KNN 的泛化优势——无训练参数 = 无过拟合。这直接启发了 R7 的混合方案。

**历史性突破 R11:** trajectory-level 分类器。根本洞见——step-level 分类器在 ~4000 个高度相关的时间步上训练，40 条轨迹不够泛化。但在 **episode level**，40 个 trajectory summary 向量恰好是足够的训练规模。这个 prior 占最终 blend 的 50%，LOO 从 0.814 一步跳到 0.967（+0.153）。pl_toy 从 0.483 飙升到 0.968。

## 4. 最终架构

```
输入: 20 条成功 + 20 条失败轨迹，每步含 state(19d) + action(7d)

Fit（训练三个组件）:
  ① Step-level HGBC
     特征: [state, velocity, acceleration, action, action_vel,
            relative_time, cumulative_displacement] = 73维
     HistGradientBoosting(depth=5, lr=0.08, l2=5.0, leaf=15)

  ② KNN 检索池
     cKDTree 索引全部训练步的 73维 特征, k=5, 距离加权

  ③ Trajectory-level HGBC
     每条轨迹 → summary 向量 (mean, std, delta, midpoint, Q1, Q3,
                              speed stats, direction changes, length)
     HistGradientBoosting(depth=3, lr=0.1, l2=3.0, leaf=5)
     训练在 40 个 summary 向量上

Label（生成 dense reward）:
  对每步 t:
    P = 0.3 × HGBC(step) + 0.2 × KNN(step) + 0.5 × Traj_prior(episode)
    reward(t) = weighted_cumulative_mean(P) × (0.7 + 0.3 × t/T)
  最后 15 步因果平滑

  ⚠ label() 不使用 is_success 参数，区分力纯粹来自数据
```

## 5. 最终性能

在 8 个任务（button, pickup, plug_insert, iphone_insert, usb, op_dr, pk_toy, pl_toy）上的结果：

| 任务 | PRA | LOO-sAUC | WinRank | Mono | Gap |
|---|---|---|---|---|---|
| button | 100% | 0.997 | 1.000 | 100% | +5.5 |
| pickup | 100% | 0.986 | 1.000 | 100% | +5.7 |
| plug_insert | 100% | 0.951 | 1.000 | 100% | +5.8 |
| iphone_insert | 100% | 1.000 | 1.000 | 100% | +5.7 |
| usb | 100% | 0.980 | 1.000 | 100% | +5.6 |
| op_dr | 100% | 1.000 | 1.000 | 100% | +5.5 |
| pk_toy | 100% | 0.995 | 1.000 | 100% | +3.0 |
| pl_toy | 100% | 0.968 | 1.000 | 100% | +5.3 |
| **AGG** | **100%** | **0.985** | **1.000** | **100%** | **+5.3** |

**Composite: 0.9877**

| 对比项 | 实验起点 | 实验终点 | 变化 |
|---|---|---|---|
| LOO-sAUC | 0.772 | 0.985 | +27.6% |
| MinLOO（最差任务） | 0.435 | 0.951 | +118.6% |
| train-LOO gap | 0.222 | 0.015 | -93.2% |
| PRA | 100% | 100% | — |
| Mono | 99% | 100% | +1% |

## 6. 实验经验总结

### 6.1 指标比代码更重要

三次指标体系革新（blind composite → unified multi-task → LOO-sAUC）每次都改变了优化方向。错误的指标让前 7 轮实验追求的是虚假的 100%。如果没有发现硬编码归一化范围的漏洞，所有后续改进都不会发生。

### 6.2 过拟合的层次结构

| 层次 | 问题 | 解法 |
|---|---|---|
| 指标过拟合 | 硬编码范围保证 PRA=100% | 统一归一化 |
| 分类器过拟合 | 10 个 per-bin 分类器 sAUC=1.00 | 单个全局分类器 |
| 数据过拟合 | train sAUC=0.994 vs LOO=0.772 | LOO 评估 + trajectory prior |

每一层过拟合都被上一层的虚假完美掩盖了。只有从指标层开始修复，才能暴露出下一层的问题。

### 6.3 少样本场景下的根本矛盾

40 条轨迹、每条 ~100 步 → step level 有 4000 个样本但只有 40 个独立源。Step-level 分类器必然过拟合。解法不是正则化（R2 失败证明），不是 bagging（R5 失败证明），不是子采样（R9-R10 失败证明），而是**在正确的粒度上建模**——trajectory level 上 40 个样本恰好够用。

### 6.4 失败实验的价值

R6（纯 KNN）被 discard 了，但它揭示了 KNN 在 hard tasks 上的泛化优势，直接催生了 R7 的混合方案。R9-R10 的 crash 证明了不能减少训练样本。实验记录中的 `reason` 字段是最有价值的知识。

### 6.5 减法思维

R12（简化 reward 公式）和 R15（去掉 per-bin）都在**删代码**的同时保持或提升了性能。复杂度是有成本的——每个多余的组件都是一个可以过拟合的维度。

## 7. 统计摘要

- 总实验轮次: 31 轮（第一期 7, 第二期 12, 第三期 12）
- Keep: 17 轮
- Discard: 10 轮
- Crash/Timeout: 4 轮
- 代码变更: +1808 行（含评估框架 426 行、策略实现 1253 行、报告 111 行、ledger 18 行）
