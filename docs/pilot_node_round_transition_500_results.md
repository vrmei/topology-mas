# 500 题 node-round exposure--adoption--recovery 分析

本分析遵循预先冻结的 `docs/node_round_transition_500_protocol.md`，复用既有 500 题
trace，不产生新的 LLM 调用。主分析使用固定 `T=3`；Experiment C 仅作为 horizon
敏感性分析。

## 1. 数据完整性

- 任务：GSM8K 500 题；
- 模型：Llama-3.1-8B-Instruct，temperature 0.3；
- 分层：`n=5, m in {4,8,12,16}` 与 `n=8, m in {7,10,14,21}`；
- 配对攻击条件：102,000；
- 固定 `T=3` 的 eligible benign node updates：922,500；
- 收到任意 target message 的更新：321,811；
- 收到 attack-induced target message 的更新：317,905；
- 全部新 attack-induced target adoptions：22,304；
- 在上一状态为 C 且收到 induced target 的更新中，新 induced adoption 为
  16,400 / 256,208 = 6.40%；
- 配对、消息来源、任务、图和 update key 完整性检查全部通过。

状态定义为 C（正确）、T（预先缓存的目标错误）、O（其他可解析错误）与 U（无法
解析）。U 始终单独报告。

## 2. 固定 T=3 的主结果

下表中的 exposure 是收到 attack-induced target 的更新占全部 eligible update 的比例；
adoption 是 `P(C -> T | induced target exposed)`，并排除了 paired clean update 本来也
会输出 T 的情况；recovery 是 `P(T -> C | previous attack-induced T)`。

| 分层 | induced exposure | induced adoption | induced recovery |
|---|---:|---:|---:|
| n5_m4 | 31.93% | 8.89% | 12.22% |
| n5_m8 | 43.74% | 8.07% | 13.17% |
| n5_m12 | 62.57% | 6.12% | 17.71% |
| n5_m16* | 89.46% | 3.75% | 26.21% |
| n8_m7 | 20.73% | 6.29% | 15.79% |
| n8_m10 | 24.03% | 7.80% | 14.27% |
| n8_m14 | 29.86% | 6.76% | 16.54% |
| n8_m21 | 39.92% | 5.19% | 20.90% |

`* n5_m16` 只有一张唯一完全图，不能估计同一 `(n,m)` 下的图间变异。

按“每个 task--graph--attack 条件中至少一次收到 induced target 的不同 benign
receiver 数量”衡量，错误到达范围也随当前选中图的密度增加：

| 分层 | 平均 eligible receivers | 平均 induced-target receivers | reach fraction |
|---|---:|---:|---:|
| n5_m4 | 3.250 | 1.003 | 30.85% |
| n5_m8 | 3.550 | 1.650 | 46.47% |
| n5_m12 | 4.000 | 2.754 | 68.84% |
| n5_m16* | 4.000 | 3.584 | 89.60% |
| n8_m7 | 5.629 | 0.989 | 17.56% |
| n8_m10 | 5.114 | 1.185 | 23.16% |
| n8_m14 | 5.457 | 1.665 | 30.51% |
| n8_m21 | 6.829 | 2.786 | 40.80% |

因此，在这些选中图和固定 horizon 中，density 增加确实扩大了 target error 的内部
可达范围；但到达不等于采纳。

## 3. 对核心问题的回答

结果不是“adoption suppression”与“recovery enhancement”二选一。

### 3.1 最高密度相对最低密度：两种机制同时存在

`n=5` 从 `m=4` 到 `m=16`：

- induced exposure：+57.53 pp，95% task-bootstrap CI `[+55.77,+59.19]`；
- induced adoption：-5.14 pp，`[-6.04,-4.29]`；
- induced recovery：+14.00 pp，`[+9.22,+19.20]`。

`n=8` 从 `m=7` 到 `m=21`：

- induced exposure：+19.19 pp，`[+18.60,+19.75]`；
- induced adoption：-1.10 pp，`[-1.52,-0.70]`；
- induced recovery：+5.11 pp，`[+2.87,+7.34]`。

因此，当前 500 题 trace 中最高密度端同时表现为：错误到达更多 update 和更多节点，
但正确节点在一次 exposure 后较少转为目标错误；已经形成的 attack-induced target
状态也更常恢复为正确。

而且 recovery 的增加不只是因为节点不再收到 target。在仍持续收到 target 的更新中，
`P(T -> C)` 对 `n=5` 为 11.38%、12.60%、17.31%、25.88%，对 `n=8` 为
15.29%、13.08%、15.90%、19.37%。无 target exposure 的 recovery 分母较小、区间较宽，
只作为诊断量，不据此比较密度。

### 3.2 中等密度不是单调过渡

`n=8` 从 `m=7` 到 `m=10` 时：

- induced adoption 反而增加 1.51 pp，95% CI `[+1.07,+1.97]`；
- induced recovery 减少 1.52 pp，但 CI `[-3.70,+0.59]` 包含 0。

随后 `m=10 -> 14 -> 21`，adoption 才逐段下降；recovery 到 `m=21` 才出现清楚的
上升。这个非单调形状与既有 endpoint 结果中“中间密度攻击损失较大、最高密度损失
回落”的描述一致，但 transition 分析本身尚不能证明它完全中介了 endpoint 差异。

## 4. 四类 transition

用户指定的核心 transition 在固定 `T=3` 下如下。C 行中的 T/O 是以“上一状态 C 且
本轮收到 target”为条件；T 行中的 C/T/O 是以上一状态 T 为条件。U 不合并到 O，
所以每行展示值不必相加为 1。

| 分层 | C->T | C->O | T->C | T->T | T->O |
|---|---:|---:|---:|---:|---:|
| n5_m4 | 8.96% | 1.23% | 12.62% | 76.40% | 2.80% |
| n5_m8 | 8.15% | 1.18% | 14.13% | 75.69% | 2.71% |
| n5_m12 | 6.15% | 1.16% | 18.12% | 70.72% | 3.50% |
| n5_m16* | 3.80% | 1.08% | 25.88% | 65.85% | 3.52% |
| n8_m7 | 6.35% | 1.16% | 16.39% | 72.77% | 4.09% |
| n8_m10 | 7.85% | 1.23% | 14.25% | 75.90% | 3.53% |
| n8_m14 | 6.87% | 1.18% | 17.89% | 70.77% | 5.04% |
| n8_m21 | 5.26% | 1.09% | 19.95% | 69.20% | 4.93% |

主表中的 attack-attributed adoption 略低于普通 C->T，因为它进一步排除了 clean
counterfactual 同轮也输出 T 的事件；两者趋势一致。

## 5. 轮次构成与 Experiment C

adoption 强烈依赖 round。所有分层中 Round 1 的 induced adoption 都高于 Round 2；
例如 `n8_m21` 为 6.67%、3.52%、2.65%，`n5_m16` 为 5.23%、2.65%、2.01%。
因此 aggregate density rate 同时混合了图结构与不同 round 的 exposure 构成。

按每张图的 depth 截断后，核心方向仍在最高密度端存在：

- `n=5` adoption：9.55% -> 9.46% -> 7.58% -> 4.49%；
- `n=8` adoption：7.33% -> 7.80% -> 6.76% -> 5.85%；
- `n=8` recovery：15.50% -> 14.27% -> 16.54% -> 21.34%。

但 Experiment C 同时改变 horizon、正常纠错机会和攻击持续时间，不能被解释为纯密度
干预。`n5_m16` 的 graph depth 为 1，没有后续 attack-induced T 状态可供 recovery
估计，因此 recovery 未定义。

## 6. 当前可支持的结论

在当前模型、GSM8K、固定 `T=3` 和选中图中，可以报告：

1. target exposure 随密度扩大，但 exposure 后采纳概率远小于 1；
2. 最高密度端相对最低密度端同时出现较低 adoption 与较高 recovery；
3. `n=8` 的中间密度出现相反的局部变化，因此机制不是简单的全局单调关系；
4. endpoint robustness 不能只用“错误到达多少节点”解释。

目前不能声称：

- density 对 adoption 或 recovery 具有跨图总体的因果效应；
- adoption suppression 或 recovery enhancement 单独、完全解释 endpoint paradox；
- 这些 transition 是 LLM 特有语义机制；
- 当前趋势能迁移到其他模型、任务、prompt 或全部合法图空间。

置信区间只重采样 500 个 task，条件于当前选中的少量图。下一步若要把 transition 与
readout endpoint 建立更强联系，应在相同 trace 上将 receiver 分为 readout 与内部节点，
并进行 task--graph 两级不确定性分析；这会比立即增加新的 LLM 调用更优先。

## 7. 产物

- `transition_rates.csv`：各分层 rate、计数与 task-bootstrap CI；
- `adjacent_density_contrasts.csv`：相邻密度的配对差值；
- `transition_matrix.csv`：完整 C/T/O/U transition matrix；
- `exposure_reach.csv`：每个攻击条件的 unique receiver reach；
- `transition_rates_by_round.csv`：逐轮 rate；
- `task_sufficient_statistics.csv`：可重复 bootstrap 的 task-level 充分统计量；
- `integrity_audit.json`：配对和抽取审计。
