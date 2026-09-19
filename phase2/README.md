# 第二阶段：四种训练策略对未见采集扰动的泛化

本目录独立执行第二阶段；第一阶段代码、预处理缓存、checkpoint、预测及报告只读复用。研究要求来自工作区中的《第二阶段实验方案：四种训练策略对未见采集扰动的泛化.md》。科学设置与冲突裁决见 [主配置](configs/phase2_main.yaml) 和 [预注册记录](results/logs/preregistration.json)。

## 运行

从工作区根目录 `E:/Eproject/ECG` 执行，而不是从 `phase2` 内执行。使用第一阶段已经安装的同一 Python 环境；依赖清单沿用 `phase1_ecg_robustness/requirements.txt`。训练要求可用 CUDA；本实验环境为 RTX 4070 Laptop、PyTorch 2.5.1+cu121。环境版本、设备及随机状态还保存在每个训练日志和 checkpoint 中。

```powershell
# 一次执行：冻结配置 → pilot → 技术门控 → 完整实验 → 第一阶段保护检查
.\phase1_ecg_robustness\.venv\Scripts\python.exe -u -m phase2.src.run_phase2 --stage all

# 分阶段执行；full 要求同一配置下的 pilot 技术门控已经通过
.\phase1_ecg_robustness\.venv\Scripts\python.exe -u -m phase2.src.run_phase2 --stage pilot
.\phase1_ecg_robustness\.venv\Scripts\python.exe -u -m phase2.src.run_phase2 --stage full
```

默认配置为 `phase2/configs/phase2_main.yaml`，也可显式传入 `--config`。运行入口设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，使用 FP32、确定性算法，禁用 AMP、TF32、early stopping 和学习率调度。

单独重算某一环节时，按以下顺序运行。`--stage` 均接受 `pilot` 或 `full`；噪声缓存生成器会拒绝在该阶段所有训练完成之前创建测试输入。

```powershell
$py = '.\phase1_ecg_robustness\.venv\Scripts\python.exe'
$cfg = 'phase2/configs/phase2_main.yaml'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '4'
& $py -m phase2.src.common --config $cfg --preregister
& $py -u -m phase2.src.train_phase2 --config $cfg --stage full
& $py -u -m phase2.src.generate_phase2_noise --config $cfg --stage full
& $py -u -m phase2.src.evaluate_phase2 --config $cfg --stage full
& $py -u -m phase2.src.statistics_phase2 --config $cfg --stage full
& $py -u -m phase2.src.plot_phase2 --config $cfg --stage full
& $py -m phase2.src.common --config $cfg --verify-preservation
```

科学产物生成后执行独立复核；复核不替代真实图像检查，也不重调模型或阈值：

```powershell
& $py -u -m phase2.src.verify_phase2 --config $cfg --stage pilot --device cpu
& $py -u -m phase2.src.verify_phase2 --config $cfg --stage full --device cuda
# 两阶段图像已实际逐图检查，且 visual_review.json 记录了对应文件指纹之后：
& $py -m phase2.src.report_phase2 --config $cfg
```

独立复核覆盖全部预测文件哈希、clean/主终点的 sklearn 指标重算、保存的患者抽样及显式重复记录复算、配对区间与 Holm、每个 checkpoint 的 clean/一个主终点 case 前 128 条记录推理回放，以及图源哈希、PNG 解码和 PDF 文件头。目视验收记录 `results/logs/visual_review.json` 的 `stages.pilot` / `stages.full` 分别保存 `status`、`figure_manifest` 和所查看 PNG 的 `files`（path/bytes/sha256）；只有全部实际检查后才标记 `passed`。报告入口核对当前文件与验收指纹，拒绝过期验收。

训练脚本另接受已注册网格内的 `--strategy`、`--model`、`--seed`，用于重新执行指定运行，不允许添加未注册条件。已完成训练仅在配置、数据、源文件和产物哈希吻合时复用；未完成训练从相同初始化重新执行固定轮数，不把部分 epoch 当作完成。已完成缓存及预测也核验后复用。统计脚本重新计算患者重采样结果，因此不建议为查看结果而反复执行整个流程。

原始预注册文件不能事后覆盖。训练关键源码、配置或输入身份改变时，程序拒绝沿用原实验。不得通过删除预注册来伪装已经发生的实验；新的科学设置需要独立实验目录和明确修订说明。

## 固定设计

- PTB-XL 100 Hz、12 导联、每条 1000 点；官方 folds 1–8 / 9 / 10 分别作为 train / val / test。完整队列为 17,084 / 2,146 / 2,158 条，五类多标签 NORM、MI、STTC、CD、HYP。无目标标签的 411 条记录继续排除。
- 沿用第一阶段 mV 预处理及去均值结果；不额外滤波，不逐导联归一化。先在 mV 空间加噪，再除以第一阶段完整训练集固定 RMS **0.2254905871835822 mV**；pilot 也不另估尺度。
- 原 ResNet（545,717 参数）和 TCN（134,885 参数）。AdamW，学习率 0.001、weight decay 0.0001、batch 128；无梯度裁剪，保持第一阶段实际训练规则。
- 每条训练记录每 epoch 只出现一次。增强选择互斥，不复制样本增加曝光；不同策略采用相同初始化和数据顺序规则，但增强使用独立计数器随机流，不扰动模型或 shuffle 的 RNG。

| 策略 | clean | independent-RMS | electrode |
|---|---:|---:|---:|
| clean_only | 100% | 0% | 0% |
| independent_rms | 50% | 50% | 0% |
| electrode | 50% | 0% | 50% |
| mixed | 50% | 25% | 25% |

训练噪声为 0.5–40 Hz Gaussian，20/10 dB 各半。电极传播使用同一固定导联矩阵，independent-RMS 对照逐导联匹配传播噪声能量，包括保留未受影响导联的零噪声。mixed 不叠加两种增强。训练组合和概率见 [训练组合](configs/phase2_train_electrode_combos.json) 与 YAML；十个未见组合见 [测试组合](configs/phase2_test_electrode_combos.json)。组合采用方案 §13 的明确列表，而不是将已经用于训练的单独 V1/V6 错称为未见组合。

训练噪声 base 为 10001，测试 base 为 20001–20005；另用 domain、stage、training seed、epoch、ECG ID、策略等显式键派生独立随机流。测试流不依赖模型、训练 seed、策略、SNR 或组合，因此相同电极模板可跨掩码和强度配对回放。训练和测试 base 范围分别限制在 10000–19999 和 20000–29999。

所有 epoch 均保存 checkpoint。最优 checkpoint 只由 clean-validation Macro-AUROC 选择；训练完成后，仅对该 checkpoint 的 clean-validation 概率逐类确定一次 F1 阈值。测试 clean、所有扰动、所有 SNR 共用这五个阈值；逐 epoch F1 日志使用固定 0.5，不提前调阈值。

## 运行网格

| 阶段 | 训练数据规模 | 训练 seed | 每模型轮数 | 训练运行数 | 共享测试 case 数 | 预测 NPZ 数 |
|---|---|---|---:|---:|---:|---:|
| pilot | 4000 / 1000 / 1000 | 17 | 10 | 8 | 118 | 944 |
| full | 17084 / 2146 / 2158 | 17, 29, 43, 101, 202 | 25 | 40 | 1711 | 68440 |

Pilot 评估 10/5/0 dB、十个训练组合、两个指定未见组合和全电极压力条件；其结果用于检查可执行性、曝光、强度、缓存和统计，不选择策略或改变 full 超参数。

Full Gaussian 网格包括：十个训练组合、十个未见组合、全电极；20/15/10/5/0 dB；independent-RMS / electrode / covariance 三种结构；五次固定噪声回放。15 dB 是未训练过的插值强度，5 dB 是更强的外推强度；0 dB 单列为压力测试，不进入主终点。已见组合汇总采用组合等权，不等同于训练增强的非均匀组合混合概率。

NSTDB 的 bw/ma/em、20/10/0 dB、全电极、三种结构、五次回放属于预注册的**探索性且有源域混杂的敏感性分析**。不能把导联空间记录重用作电极源后的表现解释成纯粹的真实采集机制因果证据。75% mixed 和 noisy-validation 选模这两个可选分析不执行，也不暗示其结果。

## 缓存与统计含义

测试输入采用无损的 factorized float32 表示：保存 0 dB 基噪声，同一份原始 ECG，固定 float32 SNR 系数和固定全局尺度。缓存建立时为**每个 SNR 的实际归一化输入**计算 SHA-256；评估时重建并核对相同哈希，再让全部 checkpoint 共用同一 GPU 输入张量。它不是为每个模型重新采样噪声，也不进行有损存储。

主终点为 electrode 测试噪声下，**十个未见组合 × 15/5 dB × 五次噪声**共 100 个 case 的等权 Macro-AUROC retention：先对各 checkpoint 的 case AUROC 求平均，再除以该 checkpoint 的 clean AUROC。不得先平均概率制造 ensemble。

确认性比较为 electrode−clean_only、electrode−independent_rms、mixed−electrode，两种模型共六项。各项对五个配对训练 seed 的 retention 差作双侧配对 t 检验，六项共同 Holm 校正。绝对 AUROC 差、AP、F1、drop、clean cost 及其余条件为次要或描述性结果，不再事后增加显著性检验族。

三种不确定性分开报告：

1. **训练随机性**：五次完整训练的原始值、均值、样本 SD 和 Student t 95% 区间；pilot 只有一个 seed，不生成虚假的 SD/t 区间。
2. **患者抽样**：固定 bootstrap seed 20260919，2000 次共同患者簇抽样；抽到一个患者即保留其全部 ECG 及抽样重复。对每个固定 checkpoint，以及固定训练 seed 指标的均值，计算 AUROC/逐类 AUROC、drop、retention、clean cost 和配对差的 percentile 95% CI。所有策略、条件共用同一组患者 multiplicities；这是固定模型、固定噪声条件下的区间，不是训练与患者的联合不确定性。
3. **噪声实现**：固定 checkpoint 下的五次回放均值、SD、范围和方向；噪声重复不能冒充五次额外训练。

缺类别的患者 draw 保留为未定义并计数，不补抽；Macro-AUROC 仍要求五类。无预测阳性/阴性时 PPV/NPV 不填 0，明确保存分母及定义/未定义计数。排名相关仅比较四种策略，使用平均并列排名；Spearman/Kendall 为描述性，不作正式显著性推断。未预设临床效用界值，因此不能自动把 clean 代价称为“可接受”。

## 产物索引

- `results/logs/preregistration.json`：时间戳、科学配置、组合分离、源文件及数据身份。
- `results/logs/phase1_preservation.json`：第一阶段受保护文件清单；最终复核为 `phase1_preservation_verified.json`。
- `results/logs/run_status.json`：每次阶段命令、开始/结束、退出码及日志指纹。
- `results/logs/{pilot,full}/{strategy}/{model}/seed_N/`：`epochs.csv`、每轮增强审计 NPZ、clean-validation 预测、最终 `summary.json`。
- `results/checkpoints/{pilot,full}/{strategy}/{model}/seed_N/`：每轮及 `best.pt`，包括优化器、随机状态、阈值来源和数据身份。
- `results/test_inputs/{pilot,full}/manifest.json`：case/group 清单、实际输入哈希、基噪声文件和逐记录物理诊断。
- `results/predictions/{pilot,full}/`：逐 checkpoint × case 概率、标签、ECG/patient ID、固定阈值和所有来源哈希。
- `results/tables/{pilot,full}/metrics.csv`：每个原始预测条件的完整指标；`group_seed_metrics.csv` / `seed_summary.csv` 为组内及跨训练 seed 汇总。
- `patient_ci.csv`、`patient_bootstrap/`：区间及可追溯的共同抽样、分布数组；`bootstrap_invalid_summary.csv` 显示无效 draw。
- `primary_comparisons.csv`、`paired_effects.csv`、`noise_contrasts.csv`：主比较、配对效应和噪声方向稳定性。
- `rank_seed.csv`、`rank_summary.csv`、`rank_correlations.csv`：排名原始值与描述性一致性。
- `results/figures/{pilot,full}/`：PNG + PDF 及记录源表哈希的 figure manifest；full 共 19 张、pilot 共 18 张，主图与次要补充图分层。
- `results/logs/{pilot,full}/independent_verification.json`：独立复算范围、误差和哈希检查；`results/logs/visual_review.json` 单独记录实际图像验收。
- `results/logs/full/{training_verification,evaluation_inventory_verification,point_statistics_verification}.json`：训练、完整评估网格，以及 clean/主组点统计的额外独立核对；各文件注明实际覆盖范围。
- `results/logs/regression_verification.json`：实际回归测试命令、退出码和原始输出；`results/logs/final_acceptance.json`：最终报告、验收证据、关键产物及清理记录的汇总索引。
- `results/tables/full/report_{summary,secondary,nstdb,mechanism}.csv`：最终报告使用的数值切片；`results/logs/full/report_manifest.json` 绑定报告、源表和生成程序。
- [完整实测报告](reports/phase2_final_report.md)：数据结果、七项判断、图表及推断边界。

实现校验使用 `phase2/tests` 的噪声配对/零支持、缓存精确重建、阈值与未定义指标、患者 multiplicities、配对比值和 Holm 数学边界测试。已在真实 pilot 训练记录上执行两模型 × 四策略 CUDA forward/backward/AdamW 与 clean inference smoke；这项实现证明不替代完整实验结果。
