# ReShift-STLA v0.1.0 — 故障回放检索 + 结果驱动的选择性修正

**独立实现:** 不读取 CoRSTA 或 Mask_RAFT/RAST 的任何检查点、S0、缓存或结果。唯一的外部输入是
Backbone 的 STLAformer 预训练 checkpoint 和 Base_SupportBench 共享缓存。代码层面复用了 CoRSTA v0.2.3
(Normal 锚点、CF、剂量阶梯、视图一致性)和 Mask_RAFT-STLA(Pearson 检索、回放缓存)的实现,包名改为 `reshift`。

> **验证状态(必读):** 本版本在没有 PyTorch 的环境中编写,只做了语法检查、未定义名检查和包内 import 检查。
> **没有运行过任何 torch 单元测试、合成端到端或真实数据实验。** 请先按第 6 节在 Gadi 上跑测试,报错贴回即可定位。

## 1. 方法

在冻结 STLAformer 上:
1. **误差分解:** `Y − P0 = (P_full − P0) + (Y − P_full)`,即恢复偏移 + 正常残差。
2. **参数化部分(沿用 CoRSTA v0.2.3):** Normal 锚点 `P_N = P0 + R̂_N`;CF 恢复偏移 `Δ̂`;剂量响应 `ψ_d`;视图一致性特征。
3. **故障回放检索:** 对每个查询,用 Pearson 粗检索在 Train 中取 top-k 邻居(排除时间重叠,训练查询另加 embargo),
   把**查询的掩码**(与邻居自身可用性求交)回放到邻居历史上,用同一冻结 STLA 得到 `P0_c(M_q)`;
   与预存的 `P_full_c`、`Y_c` 一起得到每个邻居的分解。
4. **结果统计(13 维,`reshift/retrieval/stats.py`):** `r_ret`、`d_ret`、`g_ret`(经验可恢复性)、离散度、
   **留一有害率**(用其余邻居的平均偏移修正某邻居,看是否变差)、与参数化量的一致性、有效邻居数、相似度、覆盖率。
5. **臂与策略(`reshift/retrieval/head.py`):** 在 Normal 锚点上选择 {拒绝, CF×各剂量, ret_res, ret_shift, ret_both, prior}。
   策略预测每个臂相对锚点的**真实增益**,评估时硬 argmax(低于 margin 则不修正)。

## 2. 阶段

```
s0 → bank → [m0] → normal → anchor_cf → [m1] → anchor_ret → anchor_control → lock → test → audit → report
```

| 阶段 | 内容 | 位置 |
|---|---|---|
| `s0` | 自训 STLA 掩码投影(移植 v0.1 bootstrap 的 S0) | 共享 |
| `bank` | `P_full` 缓存 + 训练掩码槽银行 + Val 银行 | 共享 |
| `normal` / `anchor_cf` | 同 CoRSTA v0.2.3 | 共享 |
| `anchor_ret` | 只训练融合门控 `a_r, a_d`、τ、prior 臂 | 本 run |
| `anchor_control` | 只训练臂策略(+ CF 的诊断头) | 本 run |
| `audit` | 检查点内干预:full / shuffled / cross_history / no_replay / no_retrieval | 本 run |

**共享机制:** 每个阶段的身份只哈希该阶段及之前消费的配置(`reshift/training/layout.py`)。检索配置不进入
`s0/normal/anchor_cf` 的键,所以同一种子下所有检索消融共享这三个阶段,只在 `anchor_ret` 之后分叉。
共享目录默认是 `ReShift_exps/<target>/incomplete/<dataset>/formal/seed<k>/shared/`。
训练共享阶段时会加锁,**同一种子的消融请等共享阶段训练完再提交**(用 `-W depend=afterok:`)。

## 3. 配置与消融(`configs/reshift/`)

| 文件 | 臂 | 策略特征 | 说明 |
|---|---|---|---|
| `reshift_full.yaml` | cf, ret_res, ret_shift, ret_both | 有 | 完整模型 |
| `reshift_no_retrieval.yaml` | cf | 无 | **独立训练的无检索对照**(同一策略结构) |
| `reshift_features_only.yaml` | cf | 有 | 检索只作策略特征 |
| `reshift_arms_no_features.yaml` | 全部 | 无 | 检索臂但策略看不到邻居统计 |
| `reshift_retrieval_only.yaml` | ret_* | 有 | 所有臂中去掉参数化 CF |
| `reshift_prior.yaml` | cf, prior | 无 | 掩码条件化先验(不检索),检验检索是否只是偏置 |
| `reshift_no_consistency.yaml` | 全部 | 有 | 去掉机制 A(会重训共享阶段) |
| `reshift_no_dose.yaml` | 全部 | 有 | 剂量阶梯只留满剂量(会重训共享阶段) |

关键参数:`retrieval_bank.top_k=9`、`periods=[3]`、`train_embargo_steps=288`、`mask_slots=2`;
`stage_epochs.anchor_ret=6`、`anchor_control=10`。

## 4. 输出

`ReShift_exps/<target>/incomplete/<dataset>/formal/seed<k>/<config>/`:
- `RESULTS.md`、`summary.json`、`test/s*.json`:每档严重程度的 MAE/RMSE、`per_family`、`cf_diagnostics`
  (含 `arms`:各臂选择比例/实际增益/有害比例;`retrieval_harm_bins`:按留一有害率分箱的有害比例)。
- `audit/s*.json`、`audit/summary.json`:各干预相对 full 的 MAE 变化。
- `diagnostics/m0.json`、`diagnostics/m1.json`。

打包:`bash ReShift-STLA/scripts/reshift/pack.sh`(不含 checkpoint 和银行)。

## 5. M0–M4 实验步骤(Gadi)

```bash
export SPEED_LLM_RAG_ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG
cd $SPEED_LLM_RAG_ROOT          # ReShift-STLA/ 与 Backbone/、Base_SupportBench/ 同级
export BB_SPEED=<speed 的 STLAformer best.pt>   BB_FLOW=<flow 的 STLAformer best.pt>
J=ReShift-STLA/pbs/reshift/stage.pbs
```

**M0(S0 + Val 银行 + 诊断,不训练 ReShift 模块):**
```bash
M0S=$(qsub -v TARGET=speed,BACKBONE_CHECKPOINT=$BB_SPEED,STAGE=m0 $J)
M0F=$(qsub -v TARGET=flow,BACKBONE_CHECKPOINT=$BB_FLOW,STAGE=m0 $J)
```
看 `diagnostics/m0.json`:`correlation`(d_ret~cf、r_ret~residual、g_ret~gain)、`loo_harm_auc_for_d_ret_harm`、
`relative_gain_vs_p0`、`per_family_mae_raw`。

**M1(训练共享的 normal/anchor_cf,再做 M1 诊断):**
```bash
A=$(qsub -W depend=afterok:$M0S -v TARGET=speed,BACKBONE_CHECKPOINT=$BB_SPEED,STAGE=anchor_cf $J)
qsub -W depend=afterok:$A -v TARGET=speed,BACKBONE_CHECKPOINT=$BB_SPEED,STAGE=m1 $J
# flow 同理(用 $M0F 和 $BB_FLOW)
```
S0 由第一个需要它的任务训练;依赖链保证同一个共享阶段不会被两个任务同时训练(否则会报锁冲突)。
看 `diagnostics/m1.json`:`oracle_headroom_of_retrieval_arms`(检索臂在 {Normal, Normal+CF} 之上的逐元素理论头部空间)、
`loo_harm_auc_for_cf_harm`。

**M2(无检索对照 vs 检索特征):** 共享阶段完成后
```bash
for C in reshift_no_retrieval reshift_features_only; do
  qsub -v TARGET=speed,BACKBONE_CHECKPOINT=$BB_SPEED,STAGE=all,RESHIFT_CONFIG=$SPEED_LLM_RAG_ROOT/ReShift-STLA/configs/reshift/$C.yaml $J
done
```

**M3(完整模型与检索消融):** `reshift_full`、`reshift_arms_no_features`、`reshift_retrieval_only`、`reshift_prior`,命令同上。
`STAGE=all` 已包含 `audit`。

**M4:** `SEED=11`、`SEED=12`,`DATASET=TraffiDent_Net_01..04`;以及 `reshift_no_consistency`、`reshift_no_dose`。

**建议的判断标准(启发式,由你决定):** M0 中 s60/s90 的 `d_ret~cf` 明显为正、留一有害率 AUC 明显高于 0.5、
`p0+d_ret+r_ret` 相对 `p0` 有增益;M1 中检索臂的理论头部空间不可忽略、`loo_harm_auc_for_cf_harm` 高于 0.5。
都不满足就停止,不必进入 M2。

## 6. 测试顺序(先做)

```bash
cd $SPEED_LLM_RAG_ROOT/ReShift-STLA && conda activate TFP
python -m pytest tests/reshift -q                          # 单元测试(CPU)
python tests/reshift/integration_synthetic.py --target both  # 合成端到端:all + m0 + m1 + 共享阶段复用
python scripts/reshift/run.py --stage preflight --target speed --backbone-checkpoint $BB_SPEED
```
合成端到端会在临时目录造假 STLA 和假缓存,检查:全流程产物、诊断与审计字段、第二个配置复用共享的 `normal/anchor_cf`。

## 7. 已知风险(诚实清单)

1. **未在 torch 下运行过**,可能有张量形状、dtype、设备细节问题。
2. **训练邻居过近导致策略过度乐观:** 已加 `train_embargo_steps=288`,但训练与测试查询的邻居时间距离分布仍可能不同,是首要怀疑项。
3. **Val/Test 与 Train 的 `window_start` 是否在同一时间轴未核实。** 跨 split 的重叠排除依赖它;若不是,排除会误删少量候选(不影响正确性,但要知道)。
4. **成本:** 每个训练掩码槽一个银行(约 Q×k×P×N×2 字节,Net_06 上 k=9 约 3.9 GiB,按 Mask 线日志估算),构建时间未在 ReShift 上实测。
5. **S0 用基准自带掩码训练**(与 v0.1 bootstrap 一致),不是多严重程度的结构化掩码。
6. **审计是检查点内干预,不是独立重训的对照;** 独立对照是 `reshift_no_retrieval.yaml`。
7. 单种子结论不可靠,正式结论至少 3 个种子 + 区块 bootstrap。

## 8. 文件清单

新增:`reshift/retrieval/{memory,index,bank,provider,stats,head}.py`、`reshift/stla/s0.py`、`reshift/training/layout.py`、
`reshift/diagnostics.py`、`scripts/reshift/{run.py,pack.sh}`、`configs/reshift/*.yaml`(8 个)、`pbs/reshift/{runtime.sh,stage.pbs}`、
`tests/reshift/*`。
相对 CoRSTA v0.2.3 修改:`reshift/model.py`(检索头最后构造、`anchor_ret` 阶段、`_retrieval_forward`)、
`reshift/training/{trainer,normal_anchor,evaluation}.py`、`reshift/data/dataset.py`(`family_id`、`window_start`)、`reshift/util.py`。
