# 500 题 receiver-scope 与 incoming-mixture 分析

本分析遵循预先冻结的 `docs/receiver_scope_mixture_500_protocol.md`，复用既有 500 题
trace，不调用 LLM。主分析使用固定 `T=3`；Experiment C 只作为 horizon 敏感性分析。

## 1. 完整性和指标

- 配对攻击条件：102,000；
- eligible benign updates：922,500；
- readout updates：306,000；
- internal updates：616,500；
- exposure、adoption 和 recovery 的 scope 合计与上一轮 pooled 结果完全一致；
- trace 配对、消息来源和更新键审计全部通过。

主指标为：

- exposure：正常 receiver 当前收到至少一个 attack-induced target；
- adoption：`P(C -> attack-induced T | induced target exposed)`；
- recovery：`P(C | previous attack-induced T)`；
- persistence：`P(attack-induced T | previous attack-induced T)`。

## 2. readout 与内部节点的完整分层

### 2.1 内部节点

| 分层 | exposure | C->T | T->C | T->T |
|---|---:|---:|---:|---:|
| n5_m4 | 25.63% | 11.76% | 15.02% | 71.08% |
| n5_m8 | 51.42% | 7.47% | 17.69% | 72.76% |
| n5_m12 | 73.07% | 5.90% | 21.67% | 67.68% |
| n5_m16* | 89.44% | 4.11% | 32.10% | 61.11% |
| n8_m7 | 13.78% | 9.20% | 16.04% | 73.39% |
| n8_m10 | 23.05% | 8.36% | 16.64% | 73.46% |
| n8_m14 | 29.33% | 7.39% | 19.74% | 71.80% |
| n8_m21 | 36.31% | 5.98% | 20.46% | 69.11% |

### 2.2 Readout

| 分层 | exposure | C->T | T->C | T->T |
|---|---:|---:|---:|---:|
| n5_m4 | 39.18% | 6.67% | 10.86% | 79.04% |
| n5_m8 | 34.53% | 9.27% | 11.08% | 78.75% |
| n5_m12 | 46.82% | 6.66% | 14.52% | 75.54% |
| n5_m16* | 89.50% | 3.04% | 16.23% | 75.39% |
| n8_m7 | 36.62% | 3.82% | 15.57% | 76.86% |
| n8_m10 | 25.87% | 6.81% | 12.67% | 78.92% |
| n8_m14 | 30.98% | 5.44% | 13.76% | 76.55% |
| n8_m21 | 50.86% | 3.49% | 21.55% | 71.40% |

`* n5_m16` 只有一张唯一完全图。

最重要的结果是：pooled 现象不是只由内部节点产生。readout 的 adoption 呈现与
endpoint 相同的非单调形状：

- `n=5`：6.67% -> 9.27% -> 6.66% -> 3.04%；
- `n=8`：3.82% -> 6.81% -> 5.44% -> 3.49%。

最高密度相对相邻中等密度的 readout adoption 下降为：

- `n5_m16 - n5_m12`：-3.62 pp，95% task-bootstrap CI `[-4.33,-2.89]`；
- `n8_m21 - n8_m14`：-1.95 pp，`[-2.41,-1.49]`。

`n8_m21` 的 readout recovery 相对 `n8_m14` 增加 7.79 pp，CI `[4.29,11.50]`，
persistence 减少 5.15 pp，`[-9.73,-0.82]`。`n5_m16` 的 readout recovery 相对
`n5_m12` 只增加 1.71 pp，区间 `[-4.30,8.42]` 包含 0；该分层只有一张图且 T 状态
较少，不能声称 recovery 明显增强。

### 2.3 最终 Round 3 的 readout

只有 Round 3 的 readout 状态就是最终系统答案。Round 3 的 adoption 为：

- `n=5`：5.19%、6.43%、5.63%、2.01%；
- `n=8`：2.44%、5.28%、4.26%、2.65%。

对应 95% task-bootstrap CI 分别为：

- `n5_m16`：`[1.26,2.84]`，而 `n5_m12` 为 `[4.52,6.80]`；
- `n8_m21`：`[2.20,3.13]`，而 `n8_m14` 为 `[3.58,4.97]`。

因此，readout 在最终更新中的 adoption suppression 确实存在，而不是跨轮平均造成的
假象。Round 3 recovery 在 `n8_m21` 为 17.94% `[14.48,21.80]`，也高于 `n8_m14`
的 12.24% `[9.18,15.55]`；`n=5` 的最终 recovery 差异较弱。

## 3. Incoming information composition

在 previous state 为 C 且收到 induced target 的 update 中，计算：

```text
target share = #T / (#C + #T + #O)
```

U 不进入分母，但单独记录。Readout 的信息组成如下：

| 分层 | mean #C | mean #T | mean #O | mean #U | target share |
|---|---:|---:|---:|---:|---:|
| n5_m4 | 0.696 | 1.000 | 0.046 | 0.018 | 62.86% |
| n5_m8 | 0.524 | 1.012 | 0.028 | 0.018 | 72.40% |
| n5_m12 | 1.624 | 1.076 | 0.077 | 0.062 | 51.82% |
| n5_m16* | 2.722 | 1.090 | 0.113 | 0.075 | 28.01% |
| n8_m7 | 1.706 | 1.003 | 0.110 | 0.046 | 38.80% |
| n8_m10 | 0.825 | 1.005 | 0.046 | 0.022 | 56.43% |
| n8_m14 | 1.693 | 1.029 | 0.086 | 0.049 | 45.79% |
| n8_m21 | 2.829 | 1.075 | 0.139 | 0.088 | 29.06% |

关键点不是最高密度下 target 数量下降：mean #T 始终接近 1，甚至略增。变化来自正确
消息增长更快，导致 target 在已解析输入中的占比大幅下降。

这也复现了 readout adoption 的非单调形状：

- `n=5,m=8` 的 target share 最高，adoption 也最高；
- `n=8,m=10` 的 target share 高于 `m=7`，adoption 同样上升；
- 最高密度中正确信息快速增加，target share 与 adoption 同时降至最低附近。

## 4. Composition-only decomposition

为避免只看总体相关，使用每个 `n` 与 receiver scope 内 pooled 的
`round x exact (T,C,O,U)` transition law，然后只替换各密度真实出现的输入组成分布。

结果几乎重现全部 adoption 差值。以 readout 为例：

| 对比 | observed delta | composition-predicted delta | residual |
|---|---:|---:|---:|
| n5_m4 -> n5_m8 | +2.59 pp | +2.19 pp | +0.40 pp |
| n5_m8 -> n5_m12 | -2.61 pp | -2.44 pp | -0.17 pp |
| n5_m12 -> n5_m16 | -3.62 pp | -3.70 pp | +0.08 pp |
| n8_m7 -> n8_m10 | +2.99 pp | +2.93 pp | +0.07 pp |
| n8_m10 -> n8_m14 | -1.38 pp | -1.11 pp | -0.27 pp |
| n8_m14 -> n8_m21 | -1.95 pp | -2.17 pp | +0.22 pp |

内部节点的所有相邻 residual 也都不超过 0.34 pp 的绝对值。

这是强描述性证据：在当前 trace 中，adoption 的密度差异大部分可由 receiver 当轮看到
的 C/T/O/U 数量和 round 组成重现。它不要求假设“高密度改变了 LLM 对同一消息组合
的语义更新规则”。

但这不是正式因果中介分析：incoming composition 是 density 和前序 LLM 状态共同形成
的 post-treatment 变量，exact cells 在不同密度中的支持范围不完全相同，而且 pooled
transition law 使用了所有密度数据。当前只能说 classical information mixing 是一个
足够强的描述性候选机制，不能说它已被因果证明，也不能排除消息措辞和推理内容的作用。

## 5. 当前可支持的结论

可以报告：

1. readout 与内部节点都出现 density-dependent adoption 变化；pooled 结果不是内部节点
   的假象；
2. 最高密度端的 readout adoption 明显低于相邻中等密度，且 Round 3 结果同样成立；
3. `n=8` 的最高密度 readout 同时出现更高 recovery 和更低 persistence；
4. target 的绝对输入数量没有下降，下降的是它相对正确消息的份额；
5. exact incoming composition 加 round 的简单重加权几乎重现相邻密度的 adoption 差异。

目前不能报告：

- density 对 readout adoption 的图总体因果效应；
- information mixing 已经完整、因果性地解释 endpoint paradox；
- 观察到的 transition 规律是 LLM 特有机制；
- 当前结果可迁移到其他模型、任务、prompt 或全部合法图。

## 6. 下一步

在继续调用 LLM 前，最有价值的是把 composition explanation 与经典基线正式连接：

1. 建立只读取 `(previous state, round, #C,#T,#O,#U)` 的 categorical transition model；
2. 使用 crossed graph--task holdout 检查其对 readout adoption 的外推能力；
3. 与 DeGroot/比例阈值/多数规则比较；
4. 只有当 composition model 留下稳定 residual 时，再用 matched rationale intervention
   判断 wording 或 reasoning 是否贡献额外效应。

这样可以严格区分“经典信息混合已足够解释”与“必须引入 LLM 语义机制”。
