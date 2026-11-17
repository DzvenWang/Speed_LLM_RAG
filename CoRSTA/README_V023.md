# CoRSTA v0.2.3 — 反事实剂量阶梯 (B) + 视图一致性可靠度 (A)

在 v0.2.2 (Normal-Anchored CF) 之上新增两个**可开关**的部署机制。关闭后模型与 v0.2.2 的前向算术完全一致。
冻结的 STLAformer、Normal 锚点、CF 分支、K 视图训练、因果性约束(推理只见掩码后输入)全部保留。

> **验证状态(必读):** 本版本在没有 PyTorch 的环境中编写,只做了语法/静态检查和纯数值公式检查,
> **没有运行过任何 torch 单元测试、冒烟训练或真实数据实验。**
> 请先在 Gadi 上按第 6 节顺序跑测试和冒烟,再提交正式训练。没有任何精度结论。

## 1. 要解决的问题(来自 v0.2.2 的实测)

| 观察(Net_06, seed10, TEST) | 含义 |
|---|---|
| Candidate(无门控)在 Speed 四档都优于 Final(有门控),宏平均 1.6773 vs 1.6817 | 现有标量门控没有学到有用的东西 |
| 预测增益 / 实际增益 ≈ 0.25(Speed)、0.06–0.6(Flow) | CF 修正的收益被严重高估,轻度缺失时尤甚 |
| 候选修正有害元素比例 40–48% | 轻度缺失下修正方向接近随机 |
| Normal 在 s20/s40 帮 0.5–1.2%,在 s90 为负 | 修正量需要随缺失程度调节 |

## 2. 机制 B:反事实剂量阶梯 (Counterfactual Dose Ladder)

**教师侧**(仅 TRAIN):对每个剂量 d<1,把"原本观测到、现在被掩码"的条目按嵌套随机子集恢复比例 d,
用同一个冻结 STLA 前向得到 `P_d`(`d=0` 即 `P0`,`d=1` 即 `P_full`)。每个剂量一次额外的无梯度前向。

**学生侧**:
- 单调剂量响应 `psi_d = d ** gamma(ctx)`,`gamma = exp(clip(head(tokens), ±1.4))`,零初始化即线性。`psi_1 = 1`,对每个元素关于 d 单调递增(构造保证)。
- 响应监督:`psi_d · Δ̂_CF ≈ P_d − P0`(`dose_response_weight`,仅主视图)。
- **剂量策略头** `dose_policy`:输入 (tokens, d, `psi_d·Δ̂`, `P_N`, `|Δ̂|`, 一致性特征),预测在该剂量下相对 Normal 锚点的**真实增益**。标签为 `|P_N−Y| − |P_N+psi_d·Δ̂−Y|`(`dose_gain_weight`)。
- **部署**:每个 (sensor, horizon) 元素取预测增益最大的剂量;若最大增益 ≤ `margin` 则不修正。`final = P_N + psi_sel · Δ̂_CF`。
- anchor_control 阶段用"含显式不修正选项的 softmax 混合"(`temperature`)做端到端训练,评估一律硬 argmax。
- 它取代标量接受门控;旧的 `anchor_gain_head/anchor_acceptance_head` 仍保留和训练(诊断/兼容)。

## 3. 机制 A:视图一致性可靠度 (View-Consistency Reliability)

推理时对每个样本再生成 `views` 个更严重的视图(从**当前已观测**条目中再随机丢弃 `drop_fraction`),
无梯度地跑完整的 Normal + CF 路径。逐元素特征(`corsta_v2/counterfactual/dose.py: consistency_features`):

| # | 特征 | 含义 |
|---|---|---|
| 0 | `log1p(10·mean_k |修正后_k − 修正后_0|)` | 修正后预测的跨视图分歧 |
| 1 | `log1p(10·mean_k |Normal_k − Normal_0|)` | 未修正预测的跨视图分歧 |
| 2 | `(分歧_未修正 − 分歧_修正后)/(分歧_未修正+ε)`,裁剪到 [-3,1] | >0 表示修正把各视图拉近(可靠),<0 表示放大分歧 |
| 3 | 各视图修正量与主视图修正量同号的比例 | 方向一致性 |

特征全程 detach、只依赖掩码后输入,**不含任何标签**;评估时丢弃模式由固定种子生成,结果可复现。
它们只作为剂量策略的输入,由真实增益标签学习如何使用(不手工设阈值)。

## 4. 配置

`configs/v0_2/corsta_v023.yaml`(新增段;`stage_epochs` 同时放宽为 anchor_cf 20 / anchor_control 10,因为 v0.2.2 触顶):

```yaml
model:
  dose_ladder:   {enabled: true, levels: [0.25, 0.5, 0.75, 1.0], margin: 0.0, temperature: 0.01}
  view_consistency: {enabled: true, views: 3, drop_fraction: 0.2, seed: 10}
training:
  dose_response_weight: 0.04
  dose_gain_weight: 0.05
  dose_gain_scale: 0.1
```

消融矩阵(每个都必须用**不同的 `RUN_DIR`**,否则身份校验会拒绝):

| 配置文件 | 含义 |
|---|---|
| `corsta_v023.yaml` | A + B(完整) |
| `corsta_v023_B_only.yaml` | 只有剂量阶梯,无一致性特征 |
| `corsta_v023_A_only.yaml` | `levels:[1.0]`:二值接受/拒绝策略 + 一致性特征(无阶梯) |
| `corsta_v023_off.yaml` | 两个机制全关(≈ v0.2.2 前向,但训练轮数同 v0.2.3),作公平对照 |

注意:新增模块改变了 state_dict,**v0.2.2 的 anchor_cf/anchor_control 检查点不能复用**;`normal` 阶段因配置哈希不同也需重训。

## 5. 运行

```bash
export SPEED_LLM_RAG_ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG
export TARGET=speed BACKBONE_CHECKPOINT=... MASK_PROJ_CHECKPOINT=...
export RUN_DIR=$SPEED_LLM_RAG_ROOT/CoRSTA_v023_exps/$TARGET/incomplete/TraffiDent_Net_06/formal/seed10_AB
qsub -v TARGET,BACKBONE_CHECKPOINT,MASK_PROJ_CHECKPOINT,RUN_DIR CoRSTA/pbs/v0_2/train_v023.pbs
# 消融:额外设置 CORSTA_V2_CONFIG=$ROOT/CoRSTA/configs/v0_2/corsta_v023_B_only.yaml 与不同 RUN_DIR
```

打包诊断:`bash CoRSTA/scripts/v0_2/pack_v023.sh`。

成本:anchor_cf 阶段约 ×1.5–2(每个训练视图再做 `views` 次无梯度前向 + 每个 batch 的 3 次教师剂量前向);
想便宜一点可设 `views: 2` 或 `levels: [0.5, 1.0]`。

## 6. 测试顺序(先做)

```bash
cd CoRSTA
python -m pytest tests/v0_2/test_v023_dose_consistency.py -q          # 新增,约 20 项
python -m pytest tests/v0_2 -q                                        # 确认 v0.2.2 旧测试不回归
python tests/v0_2/integration_synthetic_v022.py --target both          # 合成 E2E(用 v0.2.2 配置,验证旧路径)
python scripts/v0_2/run.py --smoke --cpu --config configs/v0_2/corsta_v023.yaml ...  # 冒烟(新路径),参数同 integration 脚本
```

`test_v022_anchor.py::test_control_trains_only_anchor_heads` 在默认(机制关闭)模型上仍应通过。

## 7. 新增的诊断输出(写入 `test/s*.json` 与 `summary.json` 的 `cf_diagnostics`)

- `dose.mean_applied_level / reject_fraction / level_fraction`:策略实际选择的剂量分布。
- `consistency_bins`:按"一致性收缩特征"分 4 档,各档的元素占比、**全剂量候选的有害比例**和平均增益。
  这是检验机制 A 的核心图:若有害比例随收缩特征单调下降,A 成立;若各档接近,A 无效。
- 训练日志 `loss_breakdown` 新增 `dose_response / dose_gain / dose_applied_level / dose_full_actual_gain`。

## 8. 已知风险(诚实清单)

1. **未在 torch 下运行过**。可能存在张量形状/dtype 细节问题,请以第 6 节测试为准,把报错贴回来即可定位。
2. 剂量策略的增益预测若仍像 v0.2.2 的头一样过度自信,硬 argmax 会过度修正;`margin>0`(如 0.002)是第一个要扫的超参。
3. 软混合温度 `0.01` 是按归一化增益量级(约 1e-3~5e-2)估的,未调参。
4. 教师剂量用元素级随机恢复,与真实的结构化故障(整传感器/区域)不同;`nested_restore_masks` 是替换点。
5. 一致性视图用元素级随机丢弃;结构化丢弃(整传感器/时段)可能更贴近 s90 场景,是后续消融项。
6. 增益来源可能只是"更长训练"。因此必须同时跑 `corsta_v023_off.yaml`。
7. 单种子结论不可靠,正式结论至少 3 个种子 + 区块 bootstrap。

## 9. 改动文件清单

新增:`corsta_v2/counterfactual/dose.py`、`corsta_v2/mechanisms.py`、`configs/v0_2/corsta_v023*.yaml`(4 个)、
`tests/v0_2/test_v023_dose_consistency.py`、`pbs/v0_2/train_v023.pbs`、`scripts/v0_2/pack_v023.sh`、本文件。
修改:`corsta_v2/model.py`(`_normal_core` 抽取、`_view_consistency`、剂量选择)、`counterfactual/branch.py`(`dose_head/dose_policy`)、
`counterfactual/teacher.py`(`dose_levels` → `dose_p`)、`training/normal_anchor.py`(`dose_response_loss`、`_dose_gain_loss`)、
`training/trainer.py`(教师剂量接入)、`training/evaluation.py`(剂量/一致性诊断)、`scripts/v0_2/run.py`(版本、`mechanism_kwargs`)。
