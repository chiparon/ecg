# Agent Task v2：ICASSP 一晚补实验——评价协议差中差与投影 SNR 的分层复分析

## 1. 本版变更与总目标

本任务取代 `one_night_supplementary_experiment_agent_task.md` 的 v1 版本。v1 的两项科学优先级保持不变，但本版修正了四个会影响执行正确性的地方：第一，P0 的主分析范围固定为 `heldout` 组合与 15/5 dB，而不再含混地使用 `train` 组合；第二，四臂输入一致性按测试结构和训练策略分别验证，不能错误要求 E 输入与 I 输入相同；第三，P0 和 P1 均按其可用数据分层交付，避免不必要地丢弃可复核结果；第四，删除未定义估计量的分箱标准化，并将第三架构限定为纯环境清单检查。

本夜间窗口的目标是完成两个**只读复分析**：

1. **P0：训练策略 × 测试结构的配对差中差（DiD）。** 该分析回答：在未见的电极支持组合与未见强度下，electrode augmentation 相对于 independent-RMS augmentation 的 absolute macro-AUROC 差，是否依赖于测试扰动是 electrode 结构还是 independent-RMS 结构。
2. **P1：投影 SNR 几何诊断。** 该分析回答：在相同名义总 SNR 下，E、I 与五个 signed-control 模式向 Standard 导联一致子空间投入了多少噪声能量；若逐记录预测可用，再将该几何暴露与已完成的性能曲线并列展示。

> **保护的是历史工件，而不是预先保护某个结论。** 阶段一、阶段二与 signed-control 的既有文件均只读，不覆盖、不选择性删除。新增结果可以补充、缩小或修正原解释。若发现输入身份、统计实现或原结果有错误，必须单独提交纠错记录，并暂停引用受影响结果。

除第 8 节 P3 的纯静态环境清点外，本任务不训练、不微调、不加载 checkpoint 做前向推理、不重新生成模型预测、不改输入、不重选阈值、不重选 checkpoint，也不增加 p 值、检验或多重比较家族。

---

## 2. 冻结版本、已知工件与隔离目录

在当前原实验工作树中直接执行。预测 `.npz`、逐记录诊断、输入 cache、训练结果表和现有代码均应作为只读输入。先检查 `signed_control_mechanism/logs/full/publication_verification.json`，再清点本地阶段二与 signed-control 工件。每个输入只需记录原始路径、字节数、SHA-256、schema 和来源说明；不得根据报告中的四舍五入均值反推精确数值。

优先寻找下列本地来源：

```text
phase2/results/predictions/...
phase2/results/tables/full/group_seed_metrics.csv
phase2/results/tables/full/patient_bootstrap/patient_draws.npy
phase2/results/tables/full/primary_seed_noise_metrics.csv
signed_control_mechanism/tables/full/subspace_diagnostics_per_record.csv
signed-control 的 clean/noisy input cache 或可验证的诊断来源
```

所有新文件只能写入：

```text
overnight_supplements_v2/
├── freeze/
├── shared/
├── did/
├── snrp/
├── qa/
└── reports/
```

建立 `freeze/source_manifest.json`，列出所有只读来源和内容哈希。P0 与 P1 不共享临时合并表、缓存或可写目录。

---

## 3. 全局资源门与时间记录

`T+60 min` 和 `T+90 min` 是**资源与数据可用性门**，不是科学显著性门。不要预设完整分析需要多少小时，也不要把运行超过估计时间解释为科学失败。

在前 15 分钟内，对一个预冻结代表性分片测量并写入：文件读取时间、一个 macro-AUROC 复算时间、一次患者 draw 指标计算时间，以及一次逐记录投影能量扫描时间。仅根据实测吞吐决定 P0/P1 是否并行及其 CPU/I/O 配额。若文件系统成为瓶颈，P0 优先访问原始概率；P1 暂停而不是复制未验证数据。

所有日志必须注明：

```text
training_invocations = 0
inference_invocations = 0
```

P1 的几何计算可以读取原始波形或已保存 noisy input，并重新计算标量投影能量；这不属于模型前向推理。

---

## 4. P0 主分析：heldout 组合 × 15/5 dB 的 DiD

### 4.1 预冻结问题与分析范围

P0 的**唯一主范围**是：

```text
noise family: Gaussian
combo_set: heldout
SNR: 15 dB, 5 dB
architectures: ResNet, TCN
training strategies: electrode, independent_rms
test structures: electrode, independent_rms
training seeds: 17, 29, 43, 101, 202
noise bases and heldout combinations: exactly as the phase-2 group definition
metric: absolute macro-AUROC
```

这里的 `heldout` 指阶段二中未见电极支持组合。15/5 dB 是联合未见组合与未见强度的测试范围。这个选择用于直接补强“评价协议会改变方法比较”的主张，并与阶段二 `bandpass__heldout__{condition}__s{snr}` 组定义保持一致。

`train` 组合 × 15/5 dB 可在将来的独立任务中作为“已见支持组合、未见强度”的敏感性分析，但**本轮不运行**。不得先计算两个范围后按结果强弱选择正文范围。

令 \(a\) 表示架构，\(d\in\{15,5\}\) 表示 SNR，\(s\) 表示训练 seed，\(T\in\{E,I\}\) 表示测试结构，\(h\in\{E,I\}\) 表示训练策略。定义：

\[
M_{hT,a,d,s}=\operatorname{macroAUROC}(f_{h,a,s},X_T;d),
\]

\[
g_{a,d}(T,s)=M_{ET,a,d,s}-M_{IT,a,d,s},
\]

\[
\Gamma_{a,d}(s)=g_{a,d}(E,s)-g_{a,d}(I,s),
\]

\[
\widehat\Gamma_{a,d}=\frac{1}{5}\sum_s\Gamma_{a,d}(s).
\]

正的 \(\Gamma\) 表示 electrode training 相对于 independent-RMS training 的 absolute macro-AUROC 优势，在 E 测试结构中比 I 测试结构中更大。它是一个**探索性交互估计**，不是总体训练优势、临床效用或真实硬件因果效应。

### 4.2 四臂关系：必须按比较维度验证

四臂写作 \(M_{EE},M_{IE},M_{EI},M_{II}\)，第一个下标为训练策略，第二个下标为测试结构。

| 训练策略 | E 测试结构 | I 测试结构 |
|---|---|---|
| electrode | \(M_{EE}\)：输入 \(X_E\) | \(M_{EI}\)：输入 \(X_I\) |
| independent-RMS | \(M_{IE}\)：**同一个** \(X_E\) | \(M_{II}\)：**同一个** \(X_I\) |

因此，禁止使用“全部四臂 final noisy input hash 相同”这一错误门槛。必须执行以下三个不同层级的检查：

1. **四臂共同身份。** 所有四臂具有相同的 `record_id`、`patient_id`、标签、类别顺序、clean 数据身份、`combo_set=heldout`、具体 `combo_id`、名义 SNR、noise base 和患者抽样映射。
2. **同一测试结构内的输入身份。** 在 E 测试结构中，\(M_{EE}\) 与 \(M_{IE}\) 必须使用逐记录相同的 `noisy_input_sha256=X_E`；在 I 测试结构中，\(M_{EI}\) 与 \(M_{II}\) 必须使用逐记录相同的 `noisy_input_sha256=X_I`。不要求 \(X_E=X_I\)。
3. **同一训练策略内的模型身份。** \(M_{EE}\) 与 \(M_{EI}\) 必须使用同一个 electrode-training checkpoint；\(M_{IE}\) 与 \(M_{II}\) 必须使用同一个 independent-RMS-training checkpoint。不同训练策略的 checkpoint 不应相同，且不应被要求相同。

来源索引必须分别记录 `clean_input_sha256`、`noisy_input_sha256`、`checkpoint_sha256`、`prediction_sha256`。不能用一个含混的 `input identity` 字段替代上述关系。

### 4.3 P0 的三层交付政策

P0 应在 60 分钟内判定自己属于以下哪一层。三层状态必须写入 `did/status.json`。

| 状态 | 必需数据 | 允许交付 | 不能交付 |
|---|---|---|---|
| `P0-COMPLETE` | 四臂逐记录概率、标签、患者 ID、输入/模型 hash、共同患者 draws | \(\widehat\Gamma\)、5 个 seed 值与 SD、共同患者 95% CI、四臂绝对值、draw 审计 | 新 p 值或显著性措辞 |
| `P0-DESCRIPTIVE` | 经验证的未舍入 seed 级四臂 absolute macro-AUROC，且行定义、权重、combo_set 与完整 case 可审计 | \(\widehat\Gamma\)、5 个 seed 值、seed SD、四臂绝对值 | 患者 CI、患者层推断、从边际 CI 推 DiD CI |
| `P0-NO-GO` | 无法取得上述任一合法来源，或四臂关系/指标复现失败 | 缺口与纠错报告 | 任何新的 DiD 数字 |

`P0-DESCRIPTIVE` 是合格的部分交付。它可以作为补充材料的种子层描述，但不能在摘要中暗示患者层精度，也不能伪称为完整发布。

若本地存在 full `group_seed_metrics.csv`，必须先核对它确实包含未舍入的四臂 absolute macro-AUROC、`combo_set`、`group_id`、`combo_id`、SNR、训练 seed、训练策略和测试结构，并以一个预冻结样例和阶段二原报告复算相符；否则它不构成 P0-DESCRIPTIVE 的合法来源。

### 4.4 完整 P0 的计算顺序

仅对 `P0-COMPLETE` 执行 2,000 次共同患者 cluster bootstrap：

1. 每个 draw 使用阶段二原始 `patient_draws.npy` 的同一 multiplicity。患者被抽中时，包含其所有 ECG。
2. 对同一 `(architecture, snr_db, train_seed, test_structure, combo_id, noise_seed, draw)`，先用四臂中的相同患者权重计算 absolute macro-AUROC。
3. 在最低层先作训练策略差 \(g(T,s)\)，再对 heldout combinations 与 noise bases 按阶段二原权重聚合。
4. 在同一 seed 与 draw 内计算 \(\Gamma(s)\)，最后在 draw 内对五个固定训练 seeds 等权平均。
5. 某 macro-AUROC draw 缺类别时记为 `NaN`，不重抽、不填零。只用五个 seed 均有效的 draw 形成 2.5%/97.5% percentile CI，并报告有效/无效 draw 数。

点估计来自完整测试患者集，不能用 bootstrap 均值代替。seed SD 使用 `ddof=1`，与患者 CI 并列，不合并。

### 4.5 明确 QA case、字段和验收

本任务中的 “case” 不是单条 ECG。它指预冻结的评估单元：

```text
architecture × snr_db × train_seed × test_structure × combo_id × noise_seed
```

在 `did_analysis.json` 中预先写入两个 QA 单元：

- `qa_unit_A`：所有维度按字典序的第一个合法 unit；
- `qa_unit_B`：所有维度按字典序的最后一个合法 unit。

对于 `P0-COMPLETE`，还固定复算 `draw_id=0` 和 `draw_id=1999`。不得在观察结果后替换 QA unit 或 draw。

`did_summary.csv` 每行必须含以下字段：

```text
architecture, snr_db, combo_set, metric, unit,
n_records, n_patients, n_heldout_combinations, n_noise_bases,
M_EE, M_IE, M_EI, M_II,
g_E, g_I, gamma,
seed_values, seed_sd,
patient_ci_low, patient_ci_high,
n_draws, n_valid_draws, n_invalid_draws,
source_status
```

`unit` 明确为 `fraction` 或 `pp`，且表内统一。至少一个独立 QA 实现必须从原始概率、标签和患者权重重新计算四个 absolute AUROC、两个 \(g\)、一个 \(\Gamma\)，而不是再次调用生产聚合函数。数值容差 `1e-12` 仅适用于相同概率、标签、ties、macro 平均和患者权重的浮点实现；若实现不同，必须解释算子差异，而不是机械放宽阈值。

禁止新增 t、Wilcoxon、置换、sign、bootstrap p 值或 Holm/FDR。即使患者 CI 不跨零，也只能写“在固定 checkpoint、固定噪声与共同患者抽样下，条件区间未含零”。

---

## 5. P1：投影 SNR 的分层几何诊断

### 5.1 数学对象与数值边界

复用 signed-control 的 Standard 导联一致投影：

\[
P=AA^\dagger,\qquad Q=I-P.
\]

用 float64、`rcond=1e-12` 构造 \(P\)，并验证：

\[
\|P-P^\top\|_F<10^{-10},\qquad \|P^2-P\|_F<10^{-10}.
\]

对记录 \(r\) 和条件 \(c\in\{E,I,S_0,\ldots,S_4\}\)，令 clean input 为 \(x_r\)，实际最终 float32 noisy input 为 \(z_{r,c}\)，并定义：

\[
n_{r,c}=\operatorname{float64}(z_{r,c})-\operatorname{float64}(x_r).
\]

绝对投影 SNR 的正式定义为：

\[
\operatorname{SNR}_{P,r,c}=10\log_{10}\frac{\|Px_r\|_F^2}{\|Pn_{r,c}\|_F^2}.
\]

若直接投影能量或逐记录总能量可得，以下恒等式仅作一致性检验：

\[
\operatorname{SNR}_{P,r,c}=\operatorname{SNR}_{T,r,c}+10\log_{10}\frac{1-q(x_r)}{1-q(n_{r,c})}.
\]

投影分母为零时输出 `+inf`；投影分子为零时输出 `null`；任何超出预定义浮点容差的负能量、q 越界、缺主键或不一致必须进入异常表。禁止添加 \(\epsilon\)、截断 q 或把异常置零。

### 5.2 P1 的四层交付政策

P1 的几何结果与分类结果不应被错误绑定。它必须分别报告 `geometry_status` 与 `performance_overlay_status`。

| 层级 | 可用数据 | 允许交付 | 不能声称 |
|---|---|---|---|
| `P1-GEO-DIRECT` | 原 clean/noisy arrays，或可验证重建相同输入的 immutable cache | 逐记录 absolute \(\operatorname{SNR}_P\)、E/I/S 的几何分布、q/能量交叉核验 | 已完成性能关联，除非预测也可连接 |
| `P1-GEO-IDENTITY` | 完整逐记录 `q(clean)`、`q(noise)` 与实际 total SNR 或总能量 | 由恒等式得到 absolute \(\operatorname{SNR}_P\)，并明确来源 | 独立完成了波形投影复算 |
| `P1-DELTA-ONLY` | 完整逐记录 `q(clean)` 与 `q(noise)`，但无 actual total SNR/总能量 | \(\Delta_P=10\log_{10}[(1-q(x))/(1-q(n))]\) 的分布，明确命名为“投影相对总 SNR 增量” | absolute \(\operatorname{SNR}_P\) |
| `P1-OVERLAY` | 上述任一几何层 + 可按 record ID 对齐的既有 E/S/I predictions | 原 AUROC 曲线与几何分布的上下对齐图；类别分层和 mode 异质性 | 因果中介、等效比较或新患者 CI |

分类概率和患者 draws **不是**计算几何量的必要条件。它们只在 `P1-OVERLAY` 或新的患者层不确定性分析中需要。因此，缺预测不可阻断 `P1-GEO-DIRECT`、`P1-GEO-IDENTITY` 或 `P1-DELTA-ONLY`。

### 5.3 两条合法的绝对 SNR_P 路线

**路线 A：直接投影。** 读取原 clean/noisy arrays，或根据原始 cache 规则恢复完全相同的 inputs 并验证 input hashes；随后计算 \(\|Px\|_F^2\) 与 \(\|Pn\|_F^2\)。此路线不依赖旧的 q CSV，但必须记录数据路径、hash、浮点精度和重建验收。

**路线 B：完整诊断恒等式。** 读取每记录 `q(clean)`、`q(noise)` 与 actual total SNR 或 clean/noise total energy，使用恒等式计算 absolute \(\operatorname{SNR}_P\)。此路线必须诚实说明没有另做原始波形投影复核，除非路线 A 也同时完成。

若只有完整 q，则只允许 `P1-DELTA-ONLY`。该结果有几何解释价值，但必须与 absolute \(\operatorname{SNR}_P\) 分开命名、分开作图、分开写结论。

### 5.4 P1 默认交付与明确排除项

P1 的默认交付是几何分布和、若可行、与已完成 AUROC 的**并列**展示：

- 在 nominal SNR 0/5/10 dB 的每个位置保留原有 E/S/I AUROC 曲线；
- 在同一 nominal SNR 下展示 E、I、五个 signed modes 的 \(\operatorname{SNR}_P\) 或 \(\Delta_P\) 分布；
- 显示每个 signed mode，不将五个 mode 当作随机独立重复；
- 若 predictions 可用，按真实五类展示正/负例的暴露分布，并在 `record_id`、`patient_id`、model、train seed、condition、SNR、noise seed、mode 上精确连接。

**本轮明确不做任何分箱标准化、重加权 AUROC、propensity 模型、等暴露曲线或中介分解。** 这些分析需要预先定义完整的排序型 AUROC 估计量与权重函数，不能在夜间临时补充。

E/I 为与五个 S mode 并列而复制时，必须保存 `source_prediction_id`；显示复制不改变统计权重或患者数。

---

## 6. P0/P1 输出、论文状态与结论限制

### 6.1 目录与必须文件

```text
overnight_supplements_v2/
├── freeze/
│   ├── source_manifest.json
│   ├── run_config.json
│   └── throughput_benchmark.json
├── did/
│   ├── status.json
│   ├── did_analysis.json
│   ├── four_arm_alignment.parquet
│   ├── did_long.csv
│   ├── did_summary.csv
│   ├── did_draw_audit.parquet              # P0-COMPLETE only
│   ├── independent_recompute.json
│   └── no_go_gap_report.md                 # P0-NO-GO only
├── snrp/
│   ├── status.json
│   ├── snrp_analysis.json
│   ├── snr_p_per_record.parquet            # absolute SNR_P layers only
│   ├── delta_p_per_record.parquet          # P1-DELTA-ONLY only
│   ├── geometry_summary.csv
│   ├── auroc_overlay.csv                    # P1-OVERLAY only
│   ├── geometry_distribution.pdf/png/svg
│   ├── nominal_snr_overlay.pdf/png/svg      # P1-OVERLAY only
│   ├── qa.json
│   └── no_go_gap_report.md                  # no valid geometric source only
├── qa/
│   ├── did_recompute.json
│   ├── snrp_recompute.json
│   └── release_checklist.md
└── reports/
    └── overnight_supplement_final_report.md
```

### 6.2 可写进论文的状态规则

| 工作包状态 | 可写入位置 | 允许措辞 |
|---|---|---|
| `P0-COMPLETE` | 主文或补充，篇幅由论文问题重要性决定 | “exploratory DiD”；同时给出四臂、seed SD、条件患者 CI |
| `P0-DESCRIPTIVE` | 补充材料或内部待完善表 | “seed-level descriptive analysis”；不得暗示患者 CI |
| `P0-NO-GO` | 不写结果；可在内部可复现记录保留缺口 | 不产生 DiD 结论 |
| `P1-GEO-*` | 机制补充图或补充材料 | “geometry diagnostic”；不声称性能解释 |
| `P1-OVERLAY` | signed-control 机制图下的辅助面板 | “consistent/inconsistent with performance ordering”；非因果 |
| `P1-DELTA-ONLY` | 仅补充材料 | “projection-relative-to-total SNR increment”；不称为 SNR_P |

既有 signed-control 主结论可维持其原始边界：在固定线性映射、100 Hz PTB-XL 表示、合成 Gaussian 噪声和两个网络下，Standard E 相比 five signed controls 更具破坏性。P0 与 P1 的新结果可以限制这一解释，但不得被预先要求朝任何方向支持它。

---

## 7. P2：独立验收

P2 不审阅汇总表本身，而是从 P0/P1 的只读来源独立复算指定单元。P0 需独立重算 QA unit A、B，以及完整状态下的 draw 0、1999；P1 需用五条固定 record IDs 独立检查 direct/identity 能量、q 恒等式和异常代码。

`release_checklist.md` 必须逐项记录：来源路径和 SHA-256、代码版本、schema、主键连接覆盖率、输入/模型匹配规则、独立复算误差、异常行数、未执行内容和最终可发布状态。若任一新图或数值无法从原始工件独立追溯，必须从 release 移除。

所有 release checklist 项通过后，再执行一次版本控制收口：只暂存 `overnight_supplements_v2/`、本轮新增的分析脚本和必要的项目说明；不得加入原始 PTB-XL 波形、输入 cache、原始逐记录概率或其他未经审计的大型中间文件。检查暂存差异后，以清晰信息提交并推送当前工作分支。最终总结必须给出 pushed commit SHA 和 `git status` 的干净状态。

---

## 8. P3：第三架构的纯静态环境清点

本轮 P3 只允许完成公开来源与环境清点，**不允许** forward、backward、第一 epoch 计时、模型 import、GPU smoke 或训练。

可交付：

```text
overnight_supplements_v2/third_architecture/
├── gate0_manifest.json
└── gate0_status.md
```

`gate0_manifest.json` 可记录拟用 Inception1D 的公开 commit、许可证、目标拓扑、计划输入形状、计划 patient split、GPU/数据路径是否存在，以及尚未验证的状态。`gate0_status.md` 应明确写 `DEFERRED — inventory only; no model execution performed.`

任何实际 forward、epoch timing 或训练必须在单独、经过重新批准的未来任务中执行。

---

## 9. 最终返回格式

返回一段简洁、数字完整且不隐去失败的总结，包含：

1. 推送后的 commit SHA、source manifest SHA-256、代码与输出路径；
2. P0 的 `COMPLETE`、`DESCRIPTIVE` 或 `NO-GO` 状态及证据；
3. P1 的几何层与 performance overlay 层状态；
4. 若 P0 有结果，完整四臂、\(g_E\)、\(g_I\)、\(\Gamma\)、seed SD 与条件患者 CI（仅完整层）；
5. 若 P1 有结果，计算路线、absolute SNR_P 或 Delta_P 的严格名称、异常数和是否有合法性能 overlay；
6. P2 独立验收的通过/失败项；
7. 任何身份冲突、缺失文件、不可复现数值、偏离或原结论需要暂停引用的原因；
8. 一段只包含已验收数字的论文插入文字。

## References

[1]: file:///home/ubuntu/upload/pasted_content_4.txt "GPT6pro review of the overnight supplementary experiment task"
[2]: file:///home/ubuntu/upload/phase2_final_report.md "Phase-2 final report"
[3]: file:///home/ubuntu/upload/pasted_file_gqDwCr_signed_control_final_report.html "Signed-control final report"
