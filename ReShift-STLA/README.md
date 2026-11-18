# ReShift-STLA v0.2.0 — 按观测覆盖度分配的演化检索 + 故障回放 + 反事实恢复

> v0.1（冻结 STLA + Pearson 检索 + 多臂策略）的说明保留在 [README_V01.md](README_V01.md)，代码在 `reshift/`，不再维护。
> v0.2 是独立的新包 `reshift2/`，**不读取 v0.1 的任何检查点、银行或缓存**。

> **验证状态：** CPU 单元测试（`tests/reshift2`，假 STLA，21 项）通过；真实数据 CPU 冒烟（`--smoke`，24 条查询、256 条记忆）跑通训练、评估和审计。
> **还没有任何 GPU 正式实验结果，也没有精度结论。**

## 1. 设计依据（前期实验的结论）

| 已经有效的机制 | 起作用的缺失档位 | v0.2 中的对应模块 |
|---|---|---|
| 检索完整 Train 窗口的未来演化，锚定到查询最后一次观测，逐节点、逐步长重排序并含 None 选项，风险门控 | s20–s60 | `_evolution`（C_E） |
| 不完整查询 ↔ 完整记忆的对比对齐，EMA 记忆编码器 | 同上 | `losses.alignment_loss`、`momentum_update` |
| 把查询自己的掩码回放到检索到的邻居上，迁移邻居的恢复偏移 | s90 | `_replay`（C_R） |
| 查询自身恢复偏移的参数化估计（Normal 锚点 + CF） | s90 最强 | `_counterfactual`（C_S） |
| 对 STLA 做渐进式微调 | s20–s60 | `MaskAwareSTLA` + `set_train_phase` |

明确**去掉**的：区域记忆（几乎无贡献）、正常残差检索 r_ret（全面有害）、标量接受门控 / 增益头 / 多臂 argmax 策略（三个版本都没打赢"永远修正"）、剂量阶梯和视图一致性（没有证据，成本 3.5 倍）。

## 2. 方法

```
Ŷ = B + g_E(φ)·C_E + g_R(φ)·C_R + g_S(φ)·C_S
```

- **B**：掩码感知 STLAformer（预训练权重 + 零初始化的观测模式嵌入），warmup → partial（最后 2 层）→ full 渐进微调。
- **恢复偏移**：冻结教师 T（未改动的预训练 STLA）下，窗口在掩码 m 下的恢复偏移 `s(x,m) = T(x_full) − T(x⊙m, m)`。`T(x_full)` 对每条 Train 记忆只算一次并缓存（`teacher.py`）。
- **C_E 演化检索**：查询编码器给出全局时间 / 空间键，在 Train 完整窗口上各取 top-6（排除时间重叠）；把候选未来相对其历史"最后观测时刻"的演化平移到查询锚点上（完全无观测的节点改用小时-星期季节新息）；逐元素打分器 + None + 风险门控。
- **C_R 故障回放**：按"只在查询已观测位置上计算的余弦相似度"选 4 个合法 Train 窗口（与学到的检索键无关；Train 查询另加 1 天的 embargo），把**查询的掩码**套到它们的完整历史上，用 T 得到恢复偏移 `s(x_c, m_q)`，再逐元素加权、风险门控。由于掩码固定、教师冻结，这些偏移**一次预计算、所有实验共用**（`replay_bank.py`）。
- **C_S 反事实恢复**：参数化头直接预测查询自身的恢复偏移；训练时用 Train 查询的完整历史得到目标 `T(x_full,q) − T(x_obs,q)`，Huber 蒸馏。推理时不需要完整历史。
- **分配门 g(φ)**：默认 `coverage` 模式下只看与标签无关的可观测性描述 φ（节点覆盖率、图邻居覆盖率、最近缺失时长、整体覆盖率、预测步长），每个分支一个 sigmoid 门。`sum`（门恒为 1）和 `learned`（φ + 各分支统计量）是对照。
- **损失**：原始单位 MAE + 0.05·RMSE + 0.15·基线保持 + 0.05·候选效用排序 + 0.03·风险 BCE + 0.10·CF 蒸馏 + 0.10·对齐 InfoNCE。

因果性：`forward()` 只接收 `x / mask / tod / dow / window_start`；所有证据未来都来自 Train；CF 目标只在训练循环里构造（`trainer.attach_cf_target`），单元测试 `test_forward_never_reads_labels` 检查改动标签不改变预测。

## 3. 文件

```
reshift2/
  config.py      配置、消融开关、共享缓存与 STLA checkpoint 解析
  data.py        Base_SupportBench 划分与 Train 完整记忆仓库
  stla.py        MaskAwareSTLA（被微调）、FrozenTeacher（冻结）
  encoders.py    观测状态编码器（全局键 + 节点查询）
  memory.py      Train-only 全局检索索引（时间重叠排除）
  graph.py       Train-only 正向滞后统计图
  seasonal.py    完全无观测节点的季节新息参考
  teacher.py     T(x_full) 银行
  replay_bank.py 回放偏移 + CF 目标的共享缓存（每个划分一次）
  model.py       ReShiftSTLA（三个分支 + 分配门 + 评估干预）
  losses.py      训练目标与对齐损失
  trainer.py     训练 / 预测 / 覆盖度分箱诊断
scripts/reshift2/{preflight,prepare,train,evaluate}.py
configs/reshift2/reshift2.yaml        正式配置
configs/reshift2/reshift2_fast.yaml   快速筛选配置（同一模型，~1/4 数据、~1/3 轮数）
pbs/reshift2/{prepare,train}.pbs
tests/reshift2/test_reshift2.py
```

## 4. 运行

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG && conda activate TFP
export SPEED_LLM_RAG_ROOT=$PWD
python -m pytest ReShift-STLA/tests/reshift2 -q                                   # CPU 单元测试（~10 s）
python -u ReShift-STLA/scripts/reshift2/train.py --target speed --smoke            # CPU 真实数据冒烟
export BB_SPEED=$PWD/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt
export BB_FLOW=$PWD/Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt
J=ReShift-STLA/pbs/reshift2
# 1) 每个 target 只做一次：构建全部共享缓存（图、T(x_full)、train/val/test 回放和 CF 缓存）
PS=$(qsub -v TARGET=speed,BACKBONE_CHECKPOINT=$BB_SPEED $J/prepare.pbs)
PF=$(qsub -v TARGET=flow,BACKBONE_CHECKPOINT=$BB_FLOW   $J/prepare.pbs)
# 2) 计时：跑 50 步，打印 RESHIFT_PROFILE（每步数据/前向/反向时间、显存峰值、预计每轮分钟数）后退出
qsub -W depend=afterok:$PS -v TARGET=speed,BACKBONE_CHECKPOINT=$BB_SPEED,PROFILE_STEPS=50 $J/train.pbs
# 3) 快速筛选（所有消融可并行，共用缓存）
for A in full base_only no_evolution no_replay no_cf fusion_sum; do
  qsub -W depend=afterok:$PS -v TARGET=speed,BACKBONE_CHECKPOINT=$BB_SPEED,ABLATION=$A,RESHIFT2_CONFIG=$PWD/ReShift-STLA/configs/reshift2/reshift2_fast.yaml $J/train.pbs
done
# 4) 正式（默认配置；full 自动跑审计）
qsub -W depend=afterok:$PS -v TARGET=speed,BACKBONE_CHECKPOINT=$BB_SPEED $J/train.pbs
```

**排队慢时：** 一个作业里做完多件事，只排一次队。
`PREPARE=1` 先建缓存再训练；`ABLATIONS="full base_only no_evolution"` 在同一个作业里依次跑多个配置（快速配置适合这样打包）。
A100 队列：`qsub -q dgxa100 -l ncpus=16,ngpus=1,mem=64GB ...`（每块 GPU 需 16 个 CPU，单价高于 gpuvolta）。

```bash
qsub -q dgxa100 -l ncpus=16,ngpus=1,mem=64GB,walltime=03:00:00 \
  -v TARGET=speed,BACKBONE_CHECKPOINT=$BB_SPEED,PREPARE=1,PROFILE_STEPS=50 $J/train.pbs
```

消融（各自独立训练）：`ABLATION=base_only | no_evolution | no_replay | no_cf | fusion_sum | fusion_learned | no_alignment | no_risk`。
正式 `full` 训练完后自动跑审计（同一检查点）：`wrong_evidence`、`drop_evolution`、`drop_replay`、`drop_cf`、`base`；其他情况加 `--audit`。
续跑：`RESUME_CHECKPOINT=<run>/last.pt`。

输出：`ReShift_v2_exps/<target>/<mode>/<dataset>/{formal,formal_fast,profile}/seed<k>_<ablation>_<stamp>/`，
`summary.json`（含 `test_macro_MAE`）、`evaluation/severity_summary.csv`、`evaluation/s*/<mode>/summary.json`（含按节点覆盖度分箱的门均值、修正幅度、各分支边际效应）。
共享缓存：`ReShift_v2_store/<target>/<mode>/<dataset>/{train_only_graph.npz, pfull_*.npy, banks/replay_*.npz}`。

## 4.1 效率设计

| 措施 | 效果 |
|---|---|
| 回放偏移和 CF 目标预计算并共享（`prepare.py` / `replay_bank.py`） | 训练步里**不再有教师前向**（原来每步 4 次回放 + 1 次 CF）；所有消融、种子、快速/正式配置共用同一份缓存 |
| 候选打分器分解（`FactorizedScorer`） | 不再构造 `[B,C,P,N,F]` 拼接张量；隐藏宽度 48 |
| Train/Val 常驻 GPU（`DeviceBatches`） | 无 DataLoader worker；显存不够时自动退回 DataLoader |
| `train_limit` / `val_limit` / `val_every` | 快速筛选：子集训练，记忆库仍是完整 Train |
| `--profile-steps N` | 几分钟内得到每步耗时、显存峰值和预计每轮时长 |
| `runtime.deterministic: false`（快速配置） | 打开 cudnn.benchmark |
| 向量化的 wrong-evidence 采样 | 审计不再走 Python 双重循环 |

CPU 实测（真实 STLA 与数据，batch=4，一次前向+反向）：改动前 24.1 s（其中回放 11.0 s），改动后 13.2 s，回放降到 0.14 s；剩下的主要是 STLA 主干本身（前向 3.7 s）。GPU 上的数字以 `RESHIFT_PROFILE` 为准。

**V100 实测（speed，Net_06，作业 184488022）：** 缓存构建一次共 27.7 min（T(x_full) 3.5 min、train 回放+CF 17 min、val 和每档 test 各 1.4 min；磁盘 3.1 GB）。
训练 partial 阶段 batch 64：88 queries/s（每步前向 0.40 s、反向 0.33 s、数据 <1 ms），**每轮约 9.5 min**，显存峰值 19.6 GB / 32 GB。
之后又去掉了训练步里约 20 次 GPU→CPU 同步（损失在 GPU 上累计，每轮只拷一次；对齐损失改为 GPU 上查表）。

GPU 常驻显存预算（Net_06，约）：记忆仓库 1.9 GB、T(x_full) 0.47 GB、Train 回放缓存 1.86 GB + CF 0.47 GB、Train 划分 2.8 GB、Val 0.5 GB。任何一项放不下都会自动退回 CPU / mmap，并打印 `*_FALLBACK`。

## 5. 建议的实验顺序

1. `prepare` × {speed, flow}，然后 `PROFILE_STEPS=50` 看每轮耗时和显存。
2. 用 `reshift2_fast.yaml` 并行筛选 `full / base_only / no_evolution / no_replay / no_cf / fusion_sum`；有问题先在这一层改。
3. 筛选结论稳定后再跑正式配置。
4. 正式：`full` + 主消融 × {speed, flow}；`full` 补 seed 11、12；时间允许再做 `fusion_learned`、`no_alignment`、`no_risk`。
5. 其他路网（`DATASET=TraffiDent_Net_01..04`，各自先跑 `prepare`）。

**判断标准（建议）：** `full` 的四档平均 MAE 稳定低于 `base_only`，且 `no_evolution` 主要在 s20–s60 变差、`no_replay`/`no_cf` 主要在 s90 变差；`coverage` 门不差于 `sum` 和 `learned`。

## 6. 已知风险

1. 没有 GPU 实测：先跑 `PROFILE_STEPS`；显存紧张时用 `BATCH_SIZE=32` 或关掉 `runtime.gpu_dataset`。
2. 快速配置的结论只用于筛选：子集训练、轮数少，绝对数值不能写进论文。
3. 回放和 CF 用冻结教师定义恢复偏移，而基线 B 是微调后的模型；二者的偏移不完全一致，靠分配门和风险门吸收。`drop_*` 审计会显示实际贡献。
4. 训练掩码只来自基准的 s40 训练视图；s90 是分布外严重度。
5. 单种子结论不可靠；正式结论至少 3 个种子。
