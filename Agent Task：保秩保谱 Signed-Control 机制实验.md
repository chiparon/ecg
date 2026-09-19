# Agent Task：保秩保谱 Signed-Control 机制实验

## 0. 任务性质与完成定义

你将完成一项**无需重新训练**的机制消融实验，用于区分两种解释：

1. 当前观察到的鲁棒性评价差异，是否主要来自任意低秩、跨导联一致的扰动；
2. 或者，是否还取决于标准电极—导联采集映射在原始 12 导联坐标中的**特定方向和符号拓扑**。

实验必须从冻结仓库版本开始：

```text
Repository: https://github.com/chiparon/ecg
Commit: c18e9555ce8d53771167af39c784d25e7cae3445
```

**完成定义：** 产出一个独立、可复现、经过 smoke 和 full 验收的结果目录。结果中必须包含冻结配置、控制矩阵、最终输入审计、预测索引、患者级 bootstrap 摘要、子空间诊断、主图和最终报告。必须能明确支持下列两种论文解释中的一种；不得根据结果改变事前定义的符号控制或主比较。

> 若 Standard 映射噪声比 signed-control 更具破坏性，则结果支持“采集拓扑方向在低秩结构之外具有额外作用”。若两者相近，则结果支持“当前效应主要与低秩导联一致扰动有关”。两种结果都属于成功完成实验。

本任务是**探索性机制分析**。不要新增 p 值、不要创建新的多重比较显著性家族、不要把条件患者 bootstrap CI 写成跨训练随机性的总不确定性。

---

## 1. 固定科学问题与可检验假设

已有工作已比较过：

- `E`：Standard 电极映射噪声 \(n_E=A\epsilon\)；
- `I`：对每条 ECG、每条导联与 `E` RMS 匹配的独立导联噪声 \(n_I\)。

新增的 `S`（signed-control）定义为：

\[
A_S=SA,\qquad S=\operatorname{diag}(s_1,\ldots,s_{12}),\quad s_i\in\{-1,+1\},
\]

\[
n_S=A_S\epsilon=S(A\epsilon)=S n_E.
\]

clean ECG 始终不变；仅噪声残差按导联施加固定符号翻转。

令 \(C=AA^\top\)。由于 \(S\) 是正交对角矩阵，`E` 与 `S` 具有下列严格保持性质：

\[
A_SA_S^\top=SCS^\top,
\]

\[
\operatorname{rank}(A_S)=\operatorname{rank}(A),
\]

并保持协方差特征值、奇异值、每导联方差、每导联 RMS、每导联功率谱以及总能量。它改变的是跨导联协方差的符号布局，以及低维噪声子空间相对于原始导联坐标和 clean ECG 的方向。

### 1.1 主问题

在控制了低秩性质、总/逐导联噪声能量和单导联频谱后，Standard 采集映射方向是否仍比 signed-control 更容易破坏分类性能？

### 1.2 预先冻结的主比较

对 macro-AUROC，以百分点（pp）报告下列成对效应：

\[
\Delta_{E-S}=M(E)-M(\overline{S}),
\]

\[
\Delta_{S-I}=M(\overline{S})-M(I),
\]

\[
\Delta_{E-I}=M(E)-M(I).
\]

其中 \(\overline{S}\) 是固定 signed-control 模式的等权平均。负的 \(\Delta_{E-S}\) 表示 Standard 结构在相同能量、谱和秩条件下更难。

**主要终点：** 两模型的 \(\Delta_{E-S}\) 在 **0 dB** 下的 macro-AUROC 点估计、训练 seed SD 和条件患者 95% CI。

**预先规定的次要终点：**

- 5 和 10 dB 的 \(\Delta_{E-S}\)；
- 0、5、10 dB 的 \(\Delta_{S-I}\)、\(\Delta_{E-I}\)；
- macro-F1 和 ECE 的相同差异；
- signed-control 模式间的描述性差异；
- 下文定义的导联一致子空间诊断。

不应根据某一个模型、SNR 或控制模式的结果重新定义“主要比较”。

---

## 2. 不可改变的已有实验基础

### 2.1 数据与切分

复用现有 PTB-XL 处理结果与官方患者级测试划分：

- 测试集：2,158 条 ECG，1,877 名患者；
- 输入：12 导联、100 Hz、10 秒，即每条记录 `float32 [12, 1000]`，物理单位 mV；
- 类别顺序：`NORM, MI, STTC, CD, HYP`；
- 不改变样本、标签、患者 ID、测试索引、预处理或阈值。

不要上传或复制原始 PTB-XL 波形。实验仅应使用本地已有的合法数据副本，并在报告中保留官方数据来源和处理版本。

### 2.2 模型与 checkpoint

只复用既有第一阶段完整 checkpoint：

- ResNet：训练 seeds `17, 29, 43`；
- TCN：训练 seeds `17, 29, 43`；
- 推理精度：沿用原冻结环境的 deterministic FP32；关闭 AMP 和 TF32；
- batch size、归一化尺度、模型超参数、分类阈值和 checkpoint 哈希必须沿用已有记录。

**严禁重新训练、微调、重新选阈值、重新做模型选择或训练集增强搜索。**

### 2.3 噪声与 SNR

只评价：

```text
SNR: 0, 5, 10 dB
固定噪声 seeds: 8128, 101, 202, 303, 404, 505
```

每一个标准电极噪声源必须与其 signed-control 共享：

- 同一记录；
- 同一 9 个电极源实现；
- 同一 noise seed；
- 同一 Standard `E` base-noise array；
- 同一个逐记录全 12 导联 SNR 缩放因子。

对于 signed-control，必须在已生成的 Standard base-noise 上做：

```text
noise_S[lead, time] = sign[lead] * noise_E[lead, time]
```

然后采用与 Standard 完全相同的 SNR 缩放并加到同一 clean ECG：

\[
X_S=X_{clean}+\alpha_{record,SNR}\,S n_E.
\]

由于符号翻转保持范数，`E` 和每一个 `S` 的理论/数值 SNR 应相同。不得为 signed-control 重新抽样电极源或重新单独标定其 SNR。

`I` 的定义必须沿用已有的逐记录、逐导联 RMS 匹配 independent-noise 实现。由于 `S` 保持每导联 RMS，`E`、所有 `S` 与 `I` 应共享同一个 per-record/per-lead RMS 目标。

---

## 3. Signed-control 模式的冻结规则

本节必须在**任何模型预测、指标汇总或查看分类结果之前**执行并写入冻结配置。

### 3.1 模式数与生成规则

生成 **5 个**固定 signed-control 模式。五个模式是受限、平衡的机制控制，不是独立训练重复，也不作为可推广的随机矩阵总体样本。

使用确定性伪随机生成器：

```text
selection_seed = 2026092001
lead_order = [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
```

对每个候选向量 `s` 执行：

1. 固定 `s[0] = +1`，以去除整体负号的等价形式；
2. 其余 11 个元素均匀抽取 `−1/+1`；
3. 要求负号数在 4–7 之间，避免近乎全同号的退化翻转；
4. 排除全 `+1`；
5. 排除与已接受模式重复的向量；
6. 对 \(C=AA^\top\)，仅接受满足：
   \[
   \frac{\|SCS^\top-C\|_F}{\|C\|_F}>10^{-10}
   \]
   的模式；
7. 按接受顺序保留前五个模式。禁止以模型预测、AUROC、F1、ECE 或任何结果指标筛选模式。

配置中必须保存完整 `sign_vector`、候选编号、接受理由、`SCS^T` 差异范数和每个 `A_S` 的 SHA-256。最终报告和 CSV 必须列出五个模式；不可只报告平均值。

### 3.2 必需的矩阵不变量验收

对每个 signed-control 模式验证，并在 `matrix_validation.csv` 中报告：

- `rank(A_S) == rank(A)`；
- Standard 与 `A_S` 的奇异值逐项相等，最大绝对误差不高于 `1e-12`（float64）；
- `eigvals(A_S A_S^T)` 与 `eigvals(AA^T)` 相等；
- `diag(A_S A_S^T) == diag(AA^T)`；
- Standard 与 `S` 的每一导联 unit-source RMS 相同；
- \(\|SCS^T-C\|_F/\|C\|_F>10^{-10}\)；
- `A_S` 不是 `A` 的逐元素重复；
- 指出 `A_SA_S^T` 与 `AA^T` 的非对角协方差符号布局确实不同。

若任何一个模式无法通过，必须停止并修正模式选择逻辑，不能静默替换或删减。

---

## 4. 导联一致子空间诊断

### 4.1 定义

以 Standard A 定义理想导联一致子空间的正交投影：

\[
P=AA^\dagger,
\qquad Q=I_{12}-P.
\]

在 `float64` 下用 SVD/pseudoinverse 构造 \(P\)，并验证：

\[
\|P-P^\top\|_F,\; \|P^2-P\|_F < 10^{-10}.
\]

对一个记录的 12×T 波形或噪声残差 \(v\)，定义：

\[
q(v)=\frac{\|Qv\|_F^2}{\|v\|_F^2}.
\]

若分母为零，输出 `null` 并记录；不得用任意小数替代。

### 4.2 必须测量的对象

对全部测试记录计算：

1. `clean`: \(q(X_{clean})\)；
2. `E_noise`: 最终注入的 Standard 噪声残差 \(\alpha n_E\)；
3. `S_noise`: 每个 signed-control 的最终注入残差 \(\alpha S n_E\)；
4. `I_noise`: 最终注入的 independent-RMS 噪声残差；
5. `shift_noise`: 如已有 circular-shift 噪声输入和实现可直接复用，则测量它；若无法严格复用，应在报告中明确标为“不在本轮新增生成”，不能伪造或重建为等价分支；
6. 可选但推荐：`E_input`、`S_input`、`I_input`，即完整 \(X_{clean}+n\) 的 \(q\)，用于展示扰动对完整输入的影响。

对于 `E_noise`，理论上 \(q\) 应接近数值零。对于 `S_noise`，相对 **Standard 子空间** 的 \(q\) 预期为正，这是控制设计的意图；不要把它误报为实现错误。

### 4.3 子空间诊断输出

生成：

- `tables/full/subspace_diagnostics_per_record.csv`：允许大文件，但不必作为首次论文附件；
- `tables/full/subspace_diagnostics_summary.csv`：每个对象、SNR、noise seed、signed mode 的均值、SD、中位数、IQR、最小/最大值；
- `figures/full/subspace_q_distributions.pdf/png`：至少包含 clean、E、S、I 的分布；
- `figures/full/subspace_geometry.pdf/png`：显示 Standard 与 signed-control 子空间/协方差方向差异的简洁示意，不能暗示真实硬件验证。

这一诊断用于解释，不应被单独当作临床或分类性能终点。

---

## 5. 推理、统计与不确定性口径

### 5.1 计算网格

完整网格为：

```text
models: ResNet, TCN
training seeds: 17, 29, 43
conditions: E, I, S_00, S_01, S_02, S_03, S_04
SNR: 0, 5, 10 dB
noise seeds: 8128, 101, 202, 303, 404, 505
```

`E` 与 `I` 的已有预测可以复用，但必须通过以下桥接验证后才可复用：输入哈希、测试队列、checkpoint 哈希、阈值、模型版本以及既有指标需与冻结结果一致。若复用预测不可验证，则重新推理，但不得改变输入生成。

新增的 signed-control 共：

\[
5\text{ modes}\times3\text{ SNRs}\times6\text{ noise seeds}=90
\]

个噪声条件；每个模型有 3 个 checkpoint。因此，每个模型应有 270 个新增 checkpoint-condition 预测单元；两个模型共 540 个。清洁预测可复用且不应重复计入新增条件。

### 5.2 汇总顺序

对每个模型、训练 seed、SNR：

1. 对每个固定 noise seed 计算每个条件的预测指标；
2. 对每个 signed mode 与 noise seed 计算 `E−S_k`、`S_k−I`、`E−I` 的**同患者配对差**；
3. 在每个训练 seed 内对 6 个 noise seeds 和 5 个 signed modes等权平均；
4. 再对 3 个训练 seeds 等权平均得到点估计；
5. 训练不确定性报告为三 seed 的样本 SD，`ddof=1`；
6. signed mode 间变化与 noise seed 间变化单独作为描述性 SD 报告，不得伪装为额外训练重复。

### 5.3 患者级 bootstrap

- 复用冻结的患者簇重采样 draws；若必须新生成，使用 `B=2000`、固定 seed 并建立队列身份字节核验；
- 每次抽中患者时包含其所有 ECG；
- 在每一个 draw 内先计算同患者、同 checkpoint、同 noise、同 signed mode 的差，再执行上述平均；
- 取 2.5% / 97.5% percentile 形成 95% CI；
- 缺失类别 AUROC draw 保持 `NaN`，不重抽；报告无效 draw 数；
- CI 是**条件于现有 checkpoint、固定噪声及固定 signed-control 的患者不确定性**，不是训练、噪声、控制模式和患者共同变化下的总 CI。

### 5.4 指标与方向

至少计算：

- macro-AUROC；
- macro-F1：使用原 checkpoint 的冻结阈值；
- macro classwise ECE：使用原 15-bin 定义。

报告中必须明确：

- AUROC/F1 的正差值意味着第一个条件更好；
- ECE 的正差值意味着第一个条件校准更差；
- 论文主图以 AUROC 为主，F1/ECE 放到补充表或补充图，避免性能与校准方向混淆。

**不要新增 p 值、不要称显著、不要创建 Holm 家族。** 使用“点估计”“条件患者 CI”“与……一致”这类描述性语言。

---

## 6. 最终输入和跨设备验收

### 6.1 输入验收

为每个记录、SNR、noise seed、signed mode、条件生成诊断。至少验证：

| 验收项 | 条件 | 门槛 |
|---|---|---:|
| 达到的最终 SNR | E、S、I | 相对目标最大绝对误差 \(\le 10^{-4}\) dB |
| E vs S 每导联 RMS | 同记录、同 noise seed | 相对误差 \(\le 10^{-6}\) |
| E vs I 每导联 RMS | 同记录、同 noise seed | 相对误差 \(\le 10^{-6}\) |
| E vs S 每导联 periodogram | 同记录、同 noise seed | 符号翻转前后应逐频率相同；允许浮点容差 \(\le10^{-6}\) |
| E/S base-noise 对应 | 同记录、seed、mode | `S_noise == sign[:,None] * E_noise`，逐元素验证 |
| input hash | 每一最终 noisy input | 记录 SHA-256；不可覆盖已有输入 |

`S` 的功率谱保持来自“符号翻转保持每导联波形幅度”的数学性质；不要用全导联平均谱替代逐导联核验。

### 6.2 推理验收

- checkpoint SHA-256、输入 SHA-256、测试 `ids`、`patient_ids`、阈值和预测概率哈希必须在每个 `.npz` 或索引行中保存；
- 推理 worker 不得写共享可变聚合文件；每个 worker 保存自己的 ledger 与 JSON 报告；
- merge 步骤检查网格是否无重复、无缺失；
- E/I 的复用桥接应以现有 frozen 指标为准，最大概率差及标签一致率按已有 gate 验收；
- 不同设备的数值可能非位级相同，但必须报告软件/硬件版本，且不得将不同版本写成“位级可复现”。

### 6.3 运行顺序

```text
1. 创建独立 signed-control 命名空间；禁止修改 phase1、phase2、methodology_supplement 的既有科学结果。
2. 冻结 config、A、五个 sign vectors、checkpoint 列表和现有输入/预测指纹。
3. 生成并验证 signed-control base noise 和最终 noisy inputs。
4. 运行 100 条记录 smoke；完成全部结构、SNR、RMS、谱、预测网格与统计 smoke 验收。
5. 仅在 smoke 通过后运行完整 2,158 条测试集推理。
6. merge、统计、患者 bootstrap、作图、报告、publication verification。
7. 只在所有 gate 通过后标记 completed。
```

如果 smoke 失败，先修复并重跑 smoke；不得把 smoke 结果写入 full 报告，也不得跳过 smoke 直接解释 full 结果。

---

## 7. 输出目录与必须交付物

在仓库中创建一个**独立的新目录**，建议命名：

```text
signed_control_mechanism/
```

不可覆盖 `phase1_ecg_robustness/`、`phase2/` 或现有 `methodology_supplement/results/` 的任何文件。

### 7.1 必须交付的文件

```text
signed_control_mechanism/
├── configs/
│   └── signed_control_full.json
├── inputs/
│   ├── sign_controls.json
│   ├── matrices.json
│   ├── manifest.json
│   └── patient_draws provenance / hash reference
├── tables/full/
│   ├── matrix_validation.csv
│   ├── input_validation.csv
│   ├── prediction_index.csv
│   ├── signed_control_summary.csv
│   ├── signed_control_seed_effects.csv
│   ├── signed_control_noise_effects.csv
│   ├── signed_mode_effects.csv
│   ├── subspace_diagnostics_summary.csv
│   ├── subspace_diagnostics_per_record.csv
│   └── reuse_bridge.csv
├── figures/full/
│   ├── signed_control_structure_effects.pdf/png/svg
│   ├── signed_control_snr_curves.pdf/png/svg
│   ├── signed_control_matrices.pdf/png/svg
│   ├── subspace_q_distributions.pdf/png/svg
│   ├── input_validation.pdf/png/svg
│   └── manifest.json
├── logs/full/
│   ├── freeze.json
│   ├── smoke_verification.json
│   ├── full_verification.json
│   ├── publication_verification.json
│   ├── input_generation.json
│   └── worker reports
└── reports/
    ├── signed_control_final_report.html
    └── signed_control_final_report.md
```

### 7.2 最终报告的强制章节

最终 HTML 和 Markdown 报告必须包括：

1. 研究问题与预先冻结的比较；
2. Standard、signed-control、independent-RMS 的数学定义；
3. sign vectors 的冻结规则与五个具体向量；
4. 矩阵不变量及输入 RMS/SNR/谱验收；
5. 主要 AUROC 结果：0/5/10 dB、ResNet/TCN、`E−S`、`S−I`、`E−I`；
6. F1 与 ECE 的补充结果；
7. 子空间 \(q(v)\) 诊断；
8. signed mode 间的描述性变化；
9. 对两种可能结果的诚实解释；
10. 局限性与不可外推的范围；
11. 复现命令、软件/硬件环境、冻结哈希和验收状态。

报告必须明确 `signed-control` 是**保秩、保能量、保单导联谱的方向控制**，不是物理可实现的真实采集错误模型。

---

## 8. 论文解释的决策规则

### 8.1 若 Standard 比 signed-control 更难

若两个模型或至少主要模型在 0 dB 中持续出现：

\[
\Delta_{E-S}<0
\]

且条件患者 CI 与训练 seed 方向均支持稳定负向差异，则论文可写：

> 在保持扰动秩、协方差特征值、每导联能量和每导联频谱不变后，Standard 电极映射方向仍比符号重定向的低秩控制造成更大性能退化。这与采集一致的跨导联拓扑除一般低秩集中效应外还具有额外作用的解释一致。

必须保留“与……一致”“在本研究固定 A、100 Hz、合成噪声和两种网络条件下”等限定；不得声称已验证真实设备噪声或唯一因果机制。

### 8.2 若 Standard 与 signed-control 接近

若：

\[
\Delta_{E-S}\approx0
\]

或方向/模型间不稳定，则论文应写：

> 在本任务中，Standard 与保秩保谱 signed-control 的性能影响接近，提示当前结构效应主要可由低秩、导联一致的扰动性质解释；对特定采集映射方向的额外作用尚未得到支持。

这不是失败。它能防止原论文对“采集拓扑特异性”做过强表述，并使文章转向更稳健的“lead-consistent low-rank perturbation”评价框架。

### 8.3 无论结果如何都必须保留的限制

- A 是理想线性电极—导联关系，不是完整电极—皮肤界面、右腿驱动、共模抑制、运动伪影或频率依赖阻抗模型；
- signed-control 是数学机制控制，不是物理采集错误模型；
- 当前仅覆盖 PTB-XL 的既有 100 Hz 切分、ResNet 与 TCN；
- 仅覆盖 0/5/10 dB 与固定高斯电极源；
- 患者 CI 不覆盖训练与控制模式的不确定性；
- 该分析不改变第一/二阶段原预设训练比较与 Holm 结论。

---

## 9. 最终返回给主 agent 的内容

完成后，返回一段简洁但数字完整的总结，并附上：

1. 最终报告路径；
2. 最关键 CSV 路径；
3. 三张主图路径；
4. 配置 SHA-256 与代码提交哈希；
5. 是否通过 smoke、full 和 publication verification；
6. 0 dB 下 ResNet 与 TCN 的 `E−S`、`S−I`、`E−I` AUROC 点估计、seed SD 和患者 CI；
7. 5/10 dB 的对应汇总；
8. `q(clean)`、`q(E_noise)`、`q(S_noise)`、`q(I_noise)` 的主要摘要；
9. 一句话说明结果更支持“特定采集方向”还是“通用低秩导联一致扰动”；
10. 所有失败、偏离、缺失输入或不可验证的复用条件。禁止省略负面结果。

---

## References

[1]: https://github.com/chiparon/ecg/tree/c18e9555ce8d53771167af39c784d25e7cae3445 "Frozen ECG Robustness Repository"
[2]: https://physionet.org/content/ptb-xl/1.0.3/ "PTB-XL, a Large Publicly Available Electrocardiography Dataset"
[3]: https://2027.ieeeicassp.org/call-for-papers/ "Call for Papers — 2027 IEEE International Conference on Acoustics, Speech, and Signal Processing"
