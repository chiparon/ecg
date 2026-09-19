# DGX Spark 并行 Benchmark 测试方案

## 1. 测试目的

本实验不是为了证明 DGX Spark 的理论峰值算力，而是回答以下工程问题：

1. DGX Spark 能否稳定运行当前 ECG 项目的 PyTorch/CUDA 环境？
2. 与 RTX 4070 Laptop 相比，100 Hz ECG 训练和推理是否实际加速？
3. 单个训练任务运行时，GPU、CPU、内存和数据加载是否成为瓶颈？
4. 在一台 DGX Spark 上并行运行 2 个或 4 个独立 seed，是否比顺序运行更快？
5. 测试端推理、噪声生成、PSD/协方差统计是否适合 CPU/GPU 混合并行？
6. AMP/TF32 是否可以用于探索性计算，且不会改变最终确认性结果？
7. DGX Spark 是否值得用于后续 500 Hz、频域协方差匹配和多 seed 大规模实验？

本 benchmark 的最终输出不是一个“理论 TFLOPS”数字，而是：

> 针对本 ECG 项目，在固定数据、固定模型和固定实验预算下，DGX Spark 的有效吞吐、并行效率、数值一致性和迁移成本。

---

## 2. 重要原则

### 2.1 先做环境验证，再做性能测试

如果 PyTorch、CUDA、模型或数据加载没有正确工作，性能数字没有意义。必须先通过环境、单 batch 前向和小规模训练检查。

### 2.2 确认性结果与性能结果分开

性能 benchmark 可以测试 AMP、TF32、`torch.compile` 和非确定性 cuDNN，但最终论文结果仍应使用统一的确定性 FP32 配置。

### 2.3 并行粒度优先使用独立任务

如果只有一台 DGX Spark，优先比较：

- 单任务顺序运行；
- 两个独立 seed 并发；
- 四个独立 seed 并发。

不优先做多 GPU 数据并行，因为当前 ResNet/TCN 参数量较小、batch size 不大，通信成本可能超过收益。

### 2.4 所有 benchmark 都必须可复现

每次运行保存：

- Git commit 或源码 hash；
- 配置文件 hash；
- Python/PyTorch/CUDA 版本；
- GPU 与系统信息；
- 训练 seed；
- batch size；
- worker 数量；
- 精度模式；
- 起止时间；
- 峰值显存/统一内存；
- 吞吐和错误日志。

---

## 3. DGX Spark 环境自检

### 3.1 系统和架构

在 DGX Spark 上执行：

```bash
uname -a
uname -m
cat /etc/os-release
python3 --version
which python3
```

预期重点：

- Linux 系统；
- `aarch64` 或 ARM64 架构；
- Python 版本；
- 当前 Python 是否来自容器或虚拟环境。

不要从 Windows RTX 4070 Laptop 直接复制 `.venv`。DGX Spark 的 ARM64 环境应重新安装依赖或使用官方容器。

### 3.2 CUDA/PyTorch 检查

```bash
python3 - <<'PY'
import sys
import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("memory GB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY
```

同时执行设备诊断命令。不同 DGX Spark 软件版本中设备监控命令可能不同，应优先使用设备随附工具或官方文档中的命令。若 `nvidia-smi` 不可用，不要把它作为唯一失败判据。

记录到：

```text
benchmark_results/environment/environment.txt
benchmark_results/environment/torch_probe.txt
```

### 3.3 CPU、内存和磁盘

```bash
lscpu
free -h
df -h
lsblk
```

检查：

- CPU 核数；
- 统一内存大小；
- 数据盘剩余空间；
- 项目和缓存目录所在磁盘；
- 是否使用网络挂载盘。

数据必须尽量放在本地高速存储，不要把训练数据放在高延迟网络目录上作为主 benchmark。

---

## 4. 项目迁移 smoke test

### 4.1 复制最小项目内容

第一次只复制：

```text
src/
configs/phase1_smoke.yaml
configs/phase2_main.yaml
models/
一个小型 processed 数据缓存
一个 checkpoint
scripts/
tests/
requirements.txt
```

不要第一次就复制：

- 全部原始 PTB-XL；
- 全部 checkpoint；
- 所有预测 NPZ；
- 全部结果图；
- Windows 虚拟环境。

### 4.2 路径检查

```bash
grep -R "E:/\|C:/\|\\\\" -n src configs scripts || true
```

所有数据、checkpoint 和输出目录应由配置文件或环境变量指定。建议设置：

```bash
export ECG_PROJECT_ROOT=/path/to/phase2_ecg_project
export ECG_DATA_ROOT=/path/to/data
export ECG_OUTPUT_ROOT=/path/to/benchmark_results
```

### 4.3 小样本前向测试

使用 100 条测试 ECG：

- clean 输入；
- independent-RMS 输入；
- electrode 输入；
- covariance 输入；
- 100 Hz；
- batch size 8 或 16。

检查：

- 输入形状为 `(batch, 12, 1000)`；
- 输出形状为 `(batch, 5)`；
- 没有 NaN/Inf；
- checkpoint 正常加载；
- GPU 可用；
- 输出保存成功。

建议命令：

```bash
python -u -m src.dgxspark_probe \
  --config configs/phase2_main.yaml \
  --limit 100 \
  --batch-size 16 \
  --output benchmark_results/probe
```

如果项目没有 `src.dgxspark_probe`，应先使用现有 `evaluate` 入口，或新增一个仅用于 benchmark 的小脚本；不要直接在命令行粘贴大段 Python 代码。

---

## 5. 数值一致性测试

DGX Spark 和 RTX 4070 Laptop 的 GPU 架构、PyTorch/CUDA 版本可能不同，不能默认 bitwise identical。需要比较“研究结论一致性”，而非只要求每个浮点数完全相等。

### 5.1 固定测试输入

使用同一个小型输入包，包含：

- 100 条 clean ECG；
- 同一组 labels；
- 同一组 electrode noise；
- 同一组 independent-RMS noise；
- 相同 SNR；
- 相同 A 矩阵 hash；
- 相同输入文件 hash。

### 5.2 比较指标

在 RTX 4070 Laptop 和 DGX Spark 分别运行：

- logits 最大绝对差；
- 概率最大绝对差；
- Macro-AUROC；
- Macro-AP；
- Macro-F1；
- 固定阈值预测一致率；
- 电极−independent 的效应方向。

建议初步验收标准：

| 项目 | 建议标准 |
|---|---:|
| logits 最大绝对差 | 记录实际值，不预设绝对相等 |
| AUROC 差 | < 1e-5 为理想，< 1e-4 可接受 |
| AP 差 | < 1e-5 为理想，< 1e-4 可接受 |
| F1 差 | < 1e-4 为理想，< 1e-3 可接受 |
| 固定阈值标签一致率 | > 99.9% 为理想 |
| 结构效应方向 | 必须一致 |

如果差异超过标准，不要直接判定 DGX Spark 不可用。先检查：

- TF32 是否开启；
- AMP 是否开启；
- cuDNN deterministic 设置；
- PyTorch 版本；
- 输入 dtype；
- checkpoint 加载方式；
- batch size 是否改变；
- 是否存在随机噪声重新生成。

---

## 6. Benchmark 数据和模型固定配置

### 6.1 主 benchmark 数据

第一轮只使用 100 Hz 数据：

- train：4,000 条 pilot 子集；
- validation：1,000 条；
- test：1,000 条或 1,000 条固定 benchmark 子集；
- 输入长度：1000；
- 导联数：12；
- 标签数：5。

第二轮再使用 full 17,084/2,146/2,158 队列。

### 6.2 主模型

使用第一阶段完全相同的模型：

- ResNet 1D，约 545,717 参数；
- TCN，约 134,885 参数。

初期不要加入新模型，否则无法判断差异来自环境还是架构。

### 6.3 主精度配置

确认性 FP32 benchmark：

```yaml
precision: fp32
amp: false
tf32: false
deterministic: true
cudnn_benchmark: false
```

探索性性能 benchmark：

```yaml
precision: amp_fp16_or_bf16
amp: true
tf32: optional
allow_nondeterminism: true
```

两者必须分别记录，不能混在同一张速度表中。

---

## 7. Benchmark A：单任务训练基线

### 7.1 目的

测量一个真实 ECG 训练任务的：

- 总时间；
- 每 epoch 时间；
- 每 batch 时间；
- samples/s；
- GPU 利用率；
- 内存峰值；
- 数据加载等待比例。

### 7.2 设置

分别运行：

| 模型 | 数据规模 | Epoch | Seed | Batch |
|---|---:|---:|---:|---:|
| ResNet | pilot 4000 | 3 | 17 | 128 |
| TCN | pilot 4000 | 3 | 17 | 128 |
| ResNet | full 17084 | 5 | 17 | 128 |
| TCN | full 17084 | 5 | 17 | 128 |

3 epoch/5 epoch只用于 benchmark，不作为论文训练结果。每个配置至少重复 2 次，第一次包含缓存预热，第二次记录稳定吞吐。

### 7.3 记录指标

每个任务保存 JSON：

```json
{
  "host": "dgx-spark",
  "model": "resnet",
  "dataset": "pilot",
  "epochs": 3,
  "batch_size": 128,
  "num_workers": 4,
  "precision": "fp32",
  "seed": 17,
  "wall_time_sec": 0,
  "mean_epoch_sec": 0,
  "mean_step_sec": 0,
  "samples_per_sec": 0,
  "peak_memory_bytes": 0,
  "status": "passed"
}
```

### 7.4 DataLoader 消融

每个模型至少比较：

```text
num_workers = 0, 2, 4, 8
pin_memory = false, true
persistent_workers = false, true
```

不要一次组合所有参数。建议先固定 `pin_memory=true`，测试 worker 数；然后对最佳 worker 数测试 pin/persistent。

如果 DGX Spark 的统一内存架构使 `pin_memory` 没有收益，应以实际 benchmark 为准，不要照搬 RTX 4070 的最佳配置。

---

## 8. Benchmark B：batch size 和单任务吞吐

### 8.1 目的

寻找不 OOM 且吞吐最高的 batch size。

### 8.2 设置

分别对 ResNet 和 TCN 测试：

```text
batch_size = 32, 64, 128, 256, 512, 1024
```

从 128 开始向上测试；一旦 OOM，记录该配置并回退到上一个值。

### 8.3 记录

绘制：

1. batch size–samples/s 曲线；
2. batch size–wall time 曲线；
3. batch size–峰值内存曲线；
4. batch size–AUROC 数值一致性图。

训练 benchmark 的 batch size 最终只能在所有四种策略保持一致，除非报告明确把它作为性能探索而不是科学比较。

---

## 9. Benchmark C：单任务推理吞吐

### 9.1 目的

评估最小补强包和后续大规模测试端消融是否适合迁移到 DGX Spark。

### 9.2 设置

输入条件：

- clean；
- independent-RMS；
- electrode；
- covariance；
- 100 Hz；
- test 1,000 条和 full 2,158 条；
- batch size 测试 128/256/512/1024。

每个条件运行 3 次：

1. cold start；
2. warm-up 后；
3. steady-state。

排除：

- checkpoint 加载时间；
- 数据下载时间；
- 图像生成时间；
- 第一次 CUDA 初始化时间。

同时单独报告 end-to-end 时间，避免只报告理想 GPU kernel 时间。

### 9.3 关键指标

- records/s；
- batches/s；
- 单个条件总时间；
- GPU memory；
- CPU 数据准备时间；
- GPU 等待时间；
- 端到端时间。

---

## 10. Benchmark D：噪声生成和信号处理吞吐

### 10.1 目的

判断 FFT、噪声映射、协方差、Welch PSD 是否应放到 CPU 并行，或转移到 GPU。

### 10.2 子任务

分别测试：

1. 生成 independent noise；
2. 生成 electrode noise；
3. 生成 covariance-matched noise；
4. SNR 缩放；
5. 计算 12×12 covariance；
6. Welch PSD；
7. 患者 bootstrap 指标统计。

### 10.3 CPU worker 测试

```text
workers = 1, 2, 4, 8, 16
```

对每项测量：

- 处理 1,000 条记录的总时间；
- records/s；
- 峰值内存；
- CPU 利用率；
- 结果 hash。

如果 worker 数超过 CPU 物理核心或内存带宽成为瓶颈，吞吐可能下降。选择实际最快且稳定的配置。

### 10.4 CPU/GPU 分工比较

对最耗时的噪声生成任务比较：

| 方案 | 说明 |
|---|---|
| CPU single | 单进程 NumPy/SciPy |
| CPU multiprocessing | 多 worker |
| GPU torch | CUDA tensor 生成和计算 |
| pipelined | CPU 预生成 + GPU 推理 |

最推荐的候选通常是：

```text
CPU worker pool 生成噪声
GPU batch inference 推理
CPU worker pool 统计
```

但必须用实际时间验证。

---

## 11. Benchmark E：并行独立训练任务

### 11.1 目的

判断一台 DGX Spark 上同时运行多个独立 seed 是否比顺序运行更快。

### 11.2 两阶段设置

#### Pilot 并发实验

使用：

- ResNet；
- 3 epoch；
- pilot 4000 条训练数据；
- batch size 128；
- seed 17/29/43。

比较：

| 方案 | 运行方式 |
|---|---|
| S1 | 3 个任务顺序运行 |
| P2 | 2 个任务并发，第三个随后运行 |
| P3 | 3 个任务并发 |

#### Full 短跑并发实验

使用：

- ResNet 和 TCN 各 1 个任务；
- full 训练集；
- 2 epochs；
- seed 17/29。

比较：

| 方案 | 运行方式 |
|---|---|
| S2 | 两个任务顺序运行 |
| P2-full | 两个任务并发 |

### 11.3 必须记录

- 所有任务总 wall time；
- 单任务实际 wall time；
- 总吞吐；
- GPU 利用率；
- 峰值统一内存；
- OOM 或系统杀进程；
- 每个任务是否正常保存 checkpoint；
- 训练 loss/validation AUROC 是否出现异常；
- 结果 hash 是否完整。

### 11.4 并行效率

定义：

\[
Speedup(k)=\frac{T_{sequential}}{T_{parallel,k}}
\]

\[
Efficiency(k)=\frac{Speedup(k)}{k}
\]

例如：

- 2 任务并发速度提升 1.5 倍，效率为 75%；
- 4 任务并发速度提升 2.0 倍，效率为 50%。

推荐决策：

| 并行效率 | 建议 |
|---:|---|
| ≥80% | 可以采用该并行度 |
| 60–80% | 视任务规模和稳定性决定 |
| 40–60% | 通常顺序运行更稳妥 |
| <40% | 不建议并发 |

如果并发导致任何 OOM、训练中断或数值异常，即使速度更快也不得用于正式实验。

---

## 12. Benchmark F：FP32、AMP、TF32 和编译优化

### 12.1 设置

分别测试：

1. deterministic FP32；
2. non-deterministic FP32；
3. AMP FP16；
4. AMP BF16（若环境支持且稳定）；
5. TF32；
6. `torch.compile`（可选）。

### 12.2 记录

| 模式 | 训练时间 | 推理时间 | 吞吐 | 峰值内存 | AUROC | F1 | 数值差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 deterministic | | | | | | | |
| AMP | | | | | | | |
| TF32 | | | | | | | |
| compile | | | | | | | |

### 12.3 采用规则

- 最终论文确认性指标：FP32 deterministic；
- 探索筛选：可以用 AMP/TF32；
- 如果 AMP/TF32 的效应方向与 FP32 不一致，必须以 FP32 为准；
- `torch.compile` 首次编译时间单独报告；
- 如果编译后只加速短任务，不能把编译时间隐藏在结果中。

---

## 13. Benchmark G：100 Hz 与 500 Hz 扩展测试

只有在 100 Hz benchmark 通过后才执行。

### 13.1 设置

比较：

- 100 Hz、1000 点；
- 500 Hz、5000 点。

使用：

- ResNet；
- TCN；
- batch size 逐步降低；
- 2 epoch；
- 相同 seed；
- clean 和 electrode 两种输入。

### 13.2 指标

- 每 epoch wall time；
- samples/s；
- 峰值统一内存；
- 可用 batch size；
- 训练稳定性；
- 输入生成时间；
- 推理时间。

### 13.3 目的

不是立即完成 500 Hz 论文实验，而是回答：

> DGX Spark 是否足以承担后续 500 Hz 全量训练和测试。

如果 500 Hz batch size 过小或训练速度不稳定，应考虑：

- 梯度累积；
- 下采样或分段输入；
- 只在测试端使用 500 Hz；
- 租用更大 GPU，而不是强行迁移完整训练。

---

## 14. 实验目录和脚本建议

建议在项目中创建：

```text
benchmark/
├── configs/
│   ├── dgx_env.yaml
│   ├── dgx_train_single.yaml
│   ├── dgx_inference.yaml
│   ├── dgx_noise.yaml
│   ├── dgx_parallel_2.yaml
│   ├── dgx_parallel_4.yaml
│   └── dgx_precision.yaml
├── scripts/
│   ├── 00_environment_probe.sh
│   ├── 01_project_probe.sh
│   ├── 02_numeric_probe.sh
│   ├── 03_train_single.sh
│   ├── 04_batch_sweep.sh
│   ├── 05_inference_bench.sh
│   ├── 06_noise_bench.sh
│   ├── 07_parallel_bench.sh
│   ├── 08_precision_bench.sh
│   └── 09_report.sh
├── results/
│   ├── environment/
│   ├── numeric/
│   ├── train/
│   ├── inference/
│   ├── noise/
│   ├── parallel/
│   ├── precision/
│   └── figures/
└── reports/
    └── dgx_spark_benchmark_report.md
```

每个脚本必须：

- `set -euo pipefail`；
- 保存 stdout/stderr；
- 写入开始和结束时间；
- 捕获退出码；
- 保存配置 hash；
- 不覆盖已有结果；
- 失败时保留现场。

如果使用多任务并行，建议为每个任务设置独立输出目录：

```text
results/parallel/run_2_resnet_seed17/
results/parallel/run_2_resnet_seed29/
```

---

## 15. 最终验收标准

### 环境验收

- CUDA 可用；
- PyTorch 能识别 GPU；
- 模型能完成前向和反向；
- 数据和 checkpoint 可加载；
- 100 条样本 smoke test 通过。

### 数值验收

- RTX 4070 与 DGX Spark 的指标差异在可解释范围内；
- 固定测试输入和 A 矩阵 hash 一致；
- 结构效应方向一致；
- 无 NaN/Inf；
- 同一配置重复运行结果稳定。

### 性能验收

- 单任务训练完成；
- 单任务推理完成；
- batch sweep 有完整表格；
- CPU worker sweep 有完整表格；
- 并行任务无 OOM 和异常中断；
- 每个任务 checkpoint 和日志完整。

### 迁移决策

满足以下条件时，可以迁移后续大实验：

1. 小样本前向和反向稳定；
2. 与 RTX 4070 的主指标差异可接受；
3. 100 Hz 单任务比本地明显更快，或统一内存解决了 500 Hz 内存问题；
4. 至少 2 任务并行效率达到 60% 以上；
5. 数据加载不是严重瓶颈；
6. 结果目录、checkpoint 和日志管理稳定。

如果 DGX Spark 单任务没有明显加速，但 500 Hz 可以稳定运行，仍然值得用于 500 Hz 实验。如果单任务和并行都不稳定，应继续使用 RTX 4070，不要把 benchmark 失败包装成科学结果。

---

## 16. 推荐的最小执行顺序

第一天只执行以下内容：

1. 环境 probe；
2. 100 条样本前向；
3. 100 条样本反向；
4. 100 Hz pilot 3 epoch 单任务训练；
5. batch size 128/256/512；
6. 单任务推理；
7. RTX 4070 与 DGX Spark 数值对照。

如果全部通过，再执行：

8. DataLoader worker sweep；
9. CPU 噪声生成 worker sweep；
10. 两任务并发；
11. 三任务并发；
12. FP32/AMP/TF32 对照；
13. 500 Hz 2 epoch 预研。

如果第一天环境 probe 或前向失败，不进入并行 benchmark，先修复环境。

---

## 17. 最终输出报告模板

`benchmark/reports/dgx_spark_benchmark_report.md` 至少包含：

1. 硬件和软件环境；
2. 数据、模型和输入配置；
3. 数值一致性结果；
4. 单任务训练吞吐；
5. 推理吞吐；
6. batch size sweep；
7. DataLoader sweep；
8. 噪声生成和统计吞吐；
9. 单任务与并行任务对比；
10. FP32/AMP/TF32 对比；
11. 100 Hz/500 Hz 对比；
12. OOM、失败和修复记录；
13. 是否迁移后续大实验的明确决策；
14. 推荐并行度和推荐配置。

最终结论必须使用实际 wall time 和吞吐，不使用理论峰值代替实测结果。
