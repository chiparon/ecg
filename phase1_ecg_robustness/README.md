# 第一阶段：电极诱导跨导联结构扰动

验证 **electrode-induced cross-lead structured perturbation** 对 PTB-XL 五类诊断多标签模型的影响。不是从 PTB-XL 恢复真实电极电势，也不是临床设备噪声完整模型。最终结论与实测表见 [`reports/phase1_final_report.md`](reports/phase1_final_report.md)，pilot 记录见 [`reports/phase1_interim_report.md`](reports/phase1_interim_report.md)。

## 环境与数据

Python >=3.10；CPU 可运行，建议 CUDA GPU、至少 4 GB 可用磁盘（完整结果还需额外空间）。本次使用 RTX 4070 Laptop 8 GB。NVIDIA Sync 的 `dgx` DNS 解析失败，历史 IPv6 也超时，未使用远程算力。实际硬件和包版本见 `results/logs/environment.json`、`requirements-lock.txt`，尝试记录见 `results/logs/setup.log`。

通用安装（先按 PyTorch 官方方式安装适合设备的 CUDA wheel）：

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.environment
python -m pytest tests -q
python -m src.download_ptbxl --nstdb --nstdb-dir data/raw/nstdb --workers 6
python -m src.prepare_ptbxl --require-all
python -m src.prepare_ptbxl --limit 1000 --require-all --output-dir data/processed_smoke --results-dir results/smoke_preparation
python -m src.prepare_ptbxl --limit 10000 --require-all --output-dir data/processed_pilot --results-dir results/pilot_preparation
```

本机实际解释器：`E:/Eproject/ECG/phase1_ecg_robustness/.venv/Scripts/python.exe`。为不修改已有环境，本次通过已有 `inpaint` 的 Python 创建 `--system-site-packages` 虚拟环境，只读复用 Torch 2.5.1+cu121 / NumPy 2.2.6；其余依赖安装于项目虚拟环境。全新机器不需要 `inpaint`，可按上面通用流程安装。

- PTB-XL **1.0.3**：[官方页面](https://physionet.org/content/ptb-xl/1.0.3/)，CC BY 4.0；仅下载完整 **100 Hz** 波形。
- NSTDB **1.0.0**：[官方页面](https://physionet.org/content/nstdb/1.0.0/)，ODC Attribution 1.0；下载 `bw/ma/em` 原始噪声。
- 原始文件在 `data/raw/`；下载 SHA256、大小、版本、错误和文件数见 `data/download_manifest.json` 与同目录 `.inventory.jsonl`。
- 下载/预处理细节、许可、引用及外部数据目录参数见 [`data/README.md`](data/README.md)。如使用下载器默认 `data/raw/nstdb-1.0.0`，须同步更改配置 `noise.nstdb_dir`；本 README 的显式参数与已提供配置一致。
- 100 Hz 是效率选择，不能用于验证 50 Hz 以上高频噪声保真；本阶段没有 500 Hz 对照。

## 实际实验入口

所有命令从本目录执行。三个 `.sh` 是同一 Python 编排器的便捷入口，可设置 `PYTHON` 指向虚拟环境解释器；Windows 无 Bash 时直接使用等价 Python 命令。

```sh
bash scripts/run_smoke_test.sh
bash scripts/run_pilot.sh
bash scripts/run_full_phase1.sh
# 跨平台等价命令：
python -u -m src.run_experiment --config configs/phase1_smoke.yaml
python -u -m src.run_experiment --config configs/phase1_pilot.yaml
python -u -m src.run_experiment --config configs/phase1_full.yaml
```

每个入口执行：真实验证集上的噪声统计 gate → clean-only 训练 → 相同测试集配对加噪 → 统计 → 图。数值硬门槛未通过则停止；NSTDB 的 `strict_structure_control=false` 单独标记，仅允许探索性回放，不能把其模型差异纯归因于相关结构。`train --resume-completed` 仅跳过配置、数据身份和划分一致的已完成训练；不同配置必须使用新 `run_name`，不覆盖旧实验。中断训练保留最后/最佳 checkpoint，但再次运行从相同种子第一 epoch 开始，而非声称逐步恢复。评估每完成一个条件立即保存 NPZ 和汇总表；重跑评估重新计算完整配置。

数据身份还包含预处理清单、文件时间和绝对位置；重新执行 prepare 或移动已有缓存会改变身份，即使波形数值相同。要重算旧 checkpoint 的统计/图，复用已有缓存而不重复预处理；要重新预处理/迁移后训练，使用新 `run_name`。独立 evaluate 拒绝未完成 checkpoint 或数据身份不一致；statistics 拒绝未完成/缺少整个配置条件组的评估。

分别运行、重算统计或图：

```sh
python -m src.audit_noise --config configs/phase1_full.yaml
python -m src.train --config configs/phase1_full.yaml --resume-completed
python -m src.evaluate --config configs/phase1_full.yaml
python -m src.statistics --config configs/phase1_full.yaml
python -m src.plotting --config configs/phase1_full.yaml --pdf
```

额外的噪声实现敏感性已在 pilot 和 full 的原 checkpoint、原测试记录上分别执行：

```sh
python -u scripts/replay_noise.py --config configs/phase1_pilot.yaml --noise-seeds 101 202 303 404 505
python -u scripts/replay_noise.py --config configs/phase1_full.yaml --noise-seeds 101 202 303 404 505
```

该脚本重算干净预测，再对五组独立噪声基础种子执行合成带限 Gaussian 的四条件×三SNR：pilot 为180组，full 为360组。逐记录预测、强度与协议在 `results/metrics/pilot_noise_repeats/`、`results/metrics/full_noise_repeats/`，描述性差异在对应 `results/tables/` 子目录。训练种子与噪声种子是交叉设计，不把15个组合当作独立患者或15份独立训练；只报告跨噪声重复的均值、SD、范围，不另作显著性检验。

### 审阅补充实验

复用现有缓存和checkpoint，不重新训练或prepare，不覆盖既有baseline结果：

```sh
bash scripts/run_review_supplement.sh
# 跨平台等价命令：
python -u -m src.run_supplement --config configs/phase1_supplement.yaml
# 分别运行三个阶段：
python -u -m src.supplemental_audit --config configs/phase1_supplement.yaml
python -u -m src.strict_control --config configs/phase1_supplement.yaml
python -u -m src.supplemental_statistics --config configs/phase1_supplement.yaml
```

顺序为数据/采样/矩阵重放/实际注入协方差审计 → 严格边际对照推理 → 患者级AUROC bootstrap及全SNR汇总。产物单独放在 `results/{metrics,tables,figures,logs}/review_supplement/`；逐项结果见最终报告的审阅补充章节。

`marginal_shift` 对同一条electrode噪声的各导联作独立均匀循环移位，保存每个offset。它保留经验幅度多重集、方差/RMS、全记录周期谱及其频带功率，改变导联间相位/时滞对齐；**不等于独立噪声，也不声称只改变零时滞协方差**。普通有限窗口Welch估计不必一致，残差明确保存。仅对合成Gaussian做该对照，基础噪声种子8128及五个既有重放种子，覆盖两构型×三训练种子×三SNR。

新增AUROC区间使用2000次患者簇抽样，seed20260917，固定checkpoint和base8128逐记录噪声；所有条件共用患者multiplicity。每次保留被抽患者的全部ECG，重算记录级AUROC，再平均三份固定模型的AUROC；不是先平均患者概率或做模型概率集成。逐类、逐训练种子和固定三种子均值的分布均保存。它与训练种子t区间、五次噪声重放SD、原患者等权BCE区间是不同统计目标，不能互换或拼成联合CI。

## 固定设计与公平比较

- 五类顺序 `NORM, MI, STTC, CD, HYP`。使用官方 SCP diagnostic superclass 映射，按官方示例保留所有诊断代码键。**likelihood 0 代表未知，不是阴性**。无 superclass 记录显式排除。
- 官方 `strat_fold`：1–8 train、9 validation、10 test；所有患者不可跨 fold。pilot 子集种子 2026，与模型种子独立。验证集选择最佳 macro-AUROC checkpoint 和每类 F1 阈值，绝不在 noisy/test 上重调。
- 每条每导联仅去时间均值，保留 mV。训练子集计算一个全局 RMS 标量；噪声先在物理 mV 空间加入，再除同一标量。没有逐导联标准化、滤波、增强训练或裁剪。
- 模型：窄版 ResNet18 风格 1D CNN（2/2/2/2 blocks）及 dilated TCN。模型种子 17/29/43；pilot 4000/1000/1000、15 epochs；full 使用保留分析队列的全部官方分区，每架构每种子25 epochs。smoke 256/100/100、1 epoch，仅验证流程。
- full/pilot 的 `base_channels=24`；ResNet使用BatchNorm，共545717参数；TCN六个残差块通道24/24/48/48/96/96，dilation1/2/4/8/16/32、GroupNorm、dropout0.1，共134885参数。卷积部分感受野253样本，但GroupNorm和全局池化使用整段窗口，不是在线因果预测器。两构型未匹配参数量/归一化方式，不据此推断一般性架构优劣。
- Smoke 单独使用官方元数据前1000条的缓存 `data/processed_smoke`，仍按官方 fold 选择256/100/100；对应数据统计在 `results/smoke_preparation/tables/`。先行下载阶段即可运行，不与 full 缓存或数据指纹混淆。
- Pilot 使用官方元数据前10000条的独立缓存 `data/processed_pilot`，在各官方 fold 内固定选择4000/1000/1000。该候选队列并非全库均匀抽样，不能将其当成完整测试集。Full从官方21799条排除411条无目标诊断superclass记录后，按原 `strat_fold` 使用17084/2146/2158条；这是“遵循官方划分的21388条分析队列”，不是未经排除的官方原始全集。逐记录清单见 `results/tables/review_supplement/data_record_audit.csv`。
- SNR 20/10/0 dB；逐条 ECG 全12导联均方功率计算。所有测试实例保存实际 SNR、12导联 RMS/SNR 及 replay seed；模型种子不改变噪声，实现配对比较。主评估先生成 0dB 噪声后缩放，跨 SNR 复用同一波形。
- `independent`：独立导联抽样、全局 SNR 对齐；`electrode`：9节点扰动通过显式 A 映射、全局 SNR 对齐；`independent_rms`：保留独立时间实现，逐导联 RMS 精确匹配映射条件，不修改物理映射结果；`covariance`：fresh lead-space draws 经经验白化及目标协方差着色。
- 合成主噪声：矩形 FFT 带限 Gaussian，0.5–40 Hz，共同频带但并非逐频点 PSD 相等。Welch PSD/频带功率实测保存；有限记录边界具有 FFT 周期假设。
- 协方差匹配以每记录 `Cov(Aε)` 为目标，ddof=0，截断浮点舍入量级特征值，无额外 jitter；两条件通过同一**有限样本经验协方差**耦合。不据此宣称任意噪声类型的时间/高阶联合分布相同，尤其不能将 NSTDB 高阶统计也视作匹配。
- NSTDB 是**存在混杂的敏感性回放**，不与合成受控Gaussian放在同一因果证据层级。原始360Hz经SciPy1.15.3 `resample_poly(5,18)` 到100Hz：Kaiser β=5、361-tap对称FIR，1800Hz上采样网格截止50Hz，增益乘5，零填充边界及群延迟补偿；具体源码/哈希/对齐参数见 `results/tables/review_supplement/sampling_protocol.json`。它不是理想砖墙抗混叠，也不是9个真实电极的原始观测。频谱、短时相关和高阶幅度差异不能纯归因于跨导联结构。
- full 另含九个单电极 10dB 实验，并配 RMS 匹配和协方差对照。单胸前电极仅影响一个导联；在同一全局 SNR 下局部噪声被集中，敏感性不能解释为真实脱落概率/临床风险。

### A 的符号修正

电极 `[RA,LA,LL,V1..V6]`；导联 `[I,II,III,aVR,aVL,aVF,V1..V6]`。`src/lead_matrix.py` 显式定义12×9矩阵；胸前导联为 `V_i−(RA+LA+LL)/3`，所以前三系数是 **−1/3**。Guidance 示例数组中的 +1/3 与同页公式冲突，按标准公式修正。矩阵 rank=8，公共模态抵消；Einthoven、增强肢体关系、WCT、单位电极传播均有测试。可选12×10版本的 RL 列为零，不模拟右腿驱动电路动态。

审阅补充用规范JSON（schema、导联/电极顺序、数值矩阵）生成SHA256：`4f852acd9c5a4f7b35a207f5078d70f842de3077d648716950a66502951bfe9d`，固定在 `configs/phase1_supplement.yaml`。新严格对照在生成时保存矩阵哈希。旧结果没有生成时矩阵哈希，只能通过 `matrix_artifact_evidence.json`、`matrix_replay_files.csv` 做追溯验证；不向旧配置/缓存/checkpoint补写哈希，不将历史精度归档混入当前结果。

## 指标、统计与证据位置

- 主指标 macro-AUROC/AP/F1，附 micro、逐类指标、Brier、15-bin逐类平均 ECE、预测熵/置信度、clean/noisy 二值一致率、概率变化。
- 训练种子均值、样本 SD 和 Student-t 95% CI（3 seeds df=2，不能伪装大样本精度）；smoke 的单种子 SD/CI 明确为空。
- 配对 loss / probability contrasts 先跨训练种子平均，再按患者聚合；患者 bootstrap 和 sign-flip permutation，避免同患者多 ECG 或多种子伪重复。主三种噪声比较的 loss 使用一个 Holm 多重比较家族；概率与 noisy-clean 对照为探索性结果。
- 审阅新增的AUROC患者bootstrap与上述loss分析不同：按患者抽簇、保留其全部记录及抽中次数，重算记录级AUROC，再平均三个固定模型的AUROC；不平均预测概率。2000次共同抽样、base8128固定，训练种子tCI和五次噪声SD仍分别报告，不合并为联合CI。完整方法、负结果及近似控制边界见最终报告§14。
- 两架构排名仅描述性，Spearman/Kendall 只有两点时不构成普遍排名反转证据。

| 内容 | 路径 |
|---|---|
| 数据/患者/类别汇总 | `results/tables/dataset_summary.csv` |
| 数据患者泄漏/标签共现 | `results/tables/dataset_leakage.json`, `dataset_cooccurrence.json` |
| 噪声 gate/矩阵/PSD/RMS/协方差 | `results/tables/{run}/noise_gate.json`, `noise_*.csv`, `matrix_validation.json` |
| 每样本预测/噪声强度 | `results/metrics/{run}/{model}/seed_{seed}/*.npz` |
| 模型级原始指标/评估指纹 | `results/metrics/{run}/metrics.csv`, `evaluation_protocol.json` |
| 均值SD/CI、配对分析、排名 | `results/tables/{run}/metrics_summary.csv`, `paired_effects.csv`, `rank*.csv` |
| checkpoint | `results/checkpoints/{run}/{model}/seed_{seed}.pt` |
| 每epoch日志与配置 | `results/logs/{run}/` |
| 至少六类科研图与数据来源 | `results/figures/{run}/figure_manifest.json` |
| 补充患者AUROC区间/完整抽样与分布 | `results/tables/review_supplement/patient_bootstrap_auroc_ci.csv`, `patient_bootstrap_draws.npz`, `patient_bootstrap_distributions.npz` |
| 全SNR训练/噪声重放统计 | `results/tables/review_supplement/primary_three_seed_mean_sd_tci.csv`, `five_noise_replay_mean_sd.csv` |
| 新增严格边际预测/生成协议 | `results/metrics/review_supplement/strict/`, `strict_metrics.csv`, `strict_protocol.json` |
| 数据/矩阵来源/采样/协方差/边际诊断 | `results/tables/review_supplement/` |
| 补充图表及源数据清单 | `results/figures/review_supplement/figure_manifest.json`, `strict/figure_manifest.json` |
| 补充执行/回归/独立数值验收 | `results/logs/review_supplement/` |

`{run}` 为 smoke/pilot/full。NPZ 中 `p/y/ids/patient_ids/loss/thresholds/indices` 可独立复核；noisy 文件另有 `actual_snr/noise_rms/lead_snr/replay_seed`。未扰动 lead 的 SNR 为 +inf（物理上有意义）；不把它当成模型指标 NaN。原始数据/大检查点不入 Git。

训练、验证、主评估和噪声重复统一使用确定性算法，禁用 cuDNN/矩阵乘 TF32；评估协议保存实际执行标志。复核发现并修正了早期独立评估继承默认 TF32 的不一致；修正前预测、表和报告保存在 `results/logs/precision_before_fix.zip`，诊断与重算证据见 `results/logs/precision_correction.json`。最终报告只使用修正后的结果。

少数患者影响诊断同样保存在 `paired_effects.csv`：患者差异的中位数、严格大于零比例，以及去掉绝对差异最大1%/5%患者后的均值（向上取整，至少一名）。这些仅用于描述极端样本影响，不替换主估计、CI或显著性检验；患者/种子级不同统计目标不能混用。

研究限制：主实验基础噪声种子8128，pilot/full另各做五个独立重放种子；这仍固定同一测试患者队列，未构造训练/噪声/患者联合不确定性区间。只有3个训练种子、一个数据集和100Hz；未验证电极阻抗、饱和、脱落、驱动参考动态。Clean仅指未额外加噪，不保证原始记录无采集伪差。进入第二阶段前需更多独立训练、500Hz/外部数据、预注册分层推断和真实设备机制验证；不作临床部署或普遍评价偏差结论。

本次审阅补充复用六份full checkpoint，不训练或重建数据缓存；新增108份严格对照预测、6240行患者AUROC区间、11组PNG/PDF。旧1051份预测校验和保持不变，其中822份矩阵相关预测全量重放的最大概率差为0。主要补充结果与A–E逐项对应见 `reports/phase1_final_report.md` §14；执行记录、最终回归结果与独立数值复核分别为补充日志目录中的 `run_status.json`、`tests_final.xml`、`verification_numerics.json`。
