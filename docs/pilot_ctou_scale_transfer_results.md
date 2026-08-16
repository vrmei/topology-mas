# CTOU 跨系统规模迁移：n={5,8} → n={6,7}

## 实验边界

冻结的 CTOU transition table 只使用旧实验的 `n=5,8` 数据训练，并在新实验的 `n=6,7` 上测试。测试包含 127 张图和固定的 50 道 GSM8K 任务，训练与测试图无重叠。五折 task holdout 进一步排除了同一 task fold 的训练更新。

主 recursive rollout 只读取图、调度、readout、攻击节点和真实 Round-0 C/T/O/U 状态；真实 Round-1 之后的状态与 composition 不进入预测。

## 主要结果

### 局部 transition law 的跨规模迁移

Attack one-step Brier score：

| n | 内部节点 | readout |
|---:|---:|---:|
| 6 | 0.2126 | 0.1700 |
| 7 | 0.1936 | 0.1578 |

CTOU table 与 CTOU logistic 接近，且均优于 DeGroot 和 persistence。对于训练和测试规模共同覆盖、每个规模至少有 100 条样本的 transition cell，attack 条件下跨规模经验分布的加权 total variation 为 0.0101–0.0146。`n=5 vs n=8` 本身为 0.0146，因此没有观察到 `n=6,7` 特有的额外 local-law shift。

这个结果只支持当前 Llama-3.1-8B、GSM8K、T=3 和 persistent target attack 范围内的迁移，不能外推成任意规模不变性。

### Recursive rollout 的 topology-level 结果

| 层级 | n | 指标 | MAE | Spearman |
|---|---:|---|---:|---:|
| Graph | 6 | Utility | 0.0197 | 0.611 |
| Graph | 6 | Robustness | 0.0223 | 0.590 |
| Graph | 7 | Utility | 0.0245 | 0.436 |
| Graph | 7 | Robustness | 0.0203 | 0.570 |
| m-curve | 6 | Utility | 0.0116 | 0.676 |
| m-curve | 6 | Robustness | 0.0163 | 0.847 |
| m-curve | 7 | Utility | 0.0160 | 0.605 |
| m-curve | 7 | Robustness | 0.0174 | 0.874 |

Persistence 有时能接近平均 robustness 的 MAE，但不能正确排序 topology；DeGroot 对 attack robustness 的绝对值和排序均明显失配。CTOU 当前最可靠的价值是恢复 utility/robustness 的总体 topology 与 density-curve 趋势，而不是精确恢复 target-risk 和 attack-loss 的细粒度排序。

### 递归误差与支持覆盖

Attack readout 的正确概率绝对误差从 Round 1 到 Round 3 逐步增加：

| n | Round 1 | Round 2 | Round 3 |
|---:|---:|---:|---:|
| 6 | 0.0057 | 0.0097 | 0.0165 |
| 7 | 0.0063 | 0.0112 | 0.0175 |

误差表现为多轮 mean-field rollout 的渐进累积，而非 `n=7` 的突然失效。

测试更新的 exact-cell coverage 为 99.5%–99.9%，recursive rollout 的 unseen expected mass 低于 0.33%。低支持度 cell 与更大的预测误差相关，但覆盖缺口很小，不能单独解释整体迁移误差，也不能从这一相关性得出因果结论。

### Density 与规模

| n | U0 | Utility | Robustness | Target risk | ΔUtility |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.812 | 0.841 | 0.791 | 0.074 | 0.029 |
| 6 | 0.813 | 0.858 | 0.806 | 0.062 | 0.045 |
| 7 | 0.814 | 0.865 | 0.822 | 0.052 | 0.050 |
| 8 | 0.816 | 0.870 | 0.835 | 0.043 | 0.055 |

Round-0 accuracy 基本不变，但最终 utility 和 robustness 随规模提高。使用 `ρ=m/(n-1)^2` 后，不同规模的响应曲线仍有系统偏移，因此 normalized density 不能消除规模信息。

图级 Utility–Robustness Spearman 为弱正相关（0.064–0.317）；当前设置没有观察到稳定的负向 utility–robustness trade-off。

## 可支持的结论

1. `n=5,8` 学到的冻结 CTOU 在未见的 `n=6,7` 图上仍能预测局部转移，并恢复 utility/robustness 的主要曲线排序。
2. 当前误差更符合 factorized recursive rollout 的逐步累积，而不是新规模上的局部响应规律突变。
3. normalized density 不足以统一解释 `n=5–8` 的响应曲线，系统规模仍携带额外信息。

## 不能支持的结论

- 不能声称 CTOU 对任意模型、任务、攻击协议或更大规模不变。
- 不能声称 CTOU 已经是不依赖 LLM 的 topology evaluator，因为当前使用真实 Round-0 状态。
- 不能声称 support scarcity 导致误差。
- 不能声称存在 utility–robustness trade-off。
- 两批实验仍可能有 batch 差异，不能把所有规模均值差异都作因果解释。

原始分析产物保存在 `artifacts/ctou-scale-transfer-n58-to-n67-v1/`；信息边界见 `manifest.json`，核心指标见 `graph_and_curve_metrics.csv`、`round_error_summary.csv`、`same_cell_scale_metrics.csv` 和 `observed_all_size_density_curves.csv`。
