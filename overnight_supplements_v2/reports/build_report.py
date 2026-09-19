"""Render only accepted supplementary numbers and their full audit trail."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from overnight_supplements_v2.shared.common import OUT, ROOT, config, file_info, now, read_json, write_json


def rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value, digits=6, factor=1.0):
    if value is None or value == "":
        return "null"
    value = float(value) * factor
    if math.isnan(value):
        return "null"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return f"{value:.{digits}f}"


def table(headers, values):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |", *["| " + " | ".join(cell(value) for value in row) + " |" for row in values]])


def main():
    cfg = config()
    did_status = read_json(OUT / "did/status.json")
    snrp_status = read_json(OUT / "snrp/status.json")
    did_qa = read_json(OUT / "qa/did_recompute.json")
    snrp_qa = read_json(OUT / "qa/snrp_recompute.json")
    protection = read_json(OUT / "qa/source_protection.json")
    if not did_qa.get("release_pass") or snrp_qa["status"] != "passed" or protection["status"] != "passed":
        raise ValueError("Independent numerical and source protection gates must pass before report publication")
    did = rows(OUT / "did/did_summary.csv")
    if len(did) != 4 or any(row["source_status"] != "P0-COMPLETE" or row["unit"] != "fraction" for row in did):
        raise ValueError("This report release requires the observed complete four-group result, not a silent tier downgrade")
    did = sorted(did, key=lambda row: (row["architecture"], -int(row["snr_db"])))
    geometry = rows(OUT / "snrp/geometry_summary.csv")
    snr_rows = sorted([row for row in geometry if row["metric"] == "snr_p_db"], key=lambda row: (int(row["snr_db"]), row["condition"]))
    if len(snr_rows) != 21:
        raise ValueError("All seven fixed structures at all three nominal SNRs must be reported")
    source_info = file_info(OUT / "freeze/source_manifest.json")
    code_info = file_info(OUT / "freeze/code_manifest.json")
    benchmark = read_json(OUT / "freeze/throughput_benchmark.json")
    env = read_json(OUT / "freeze/environment.json")
    architecture = read_json(OUT / "third_architecture/gate0_manifest.json")
    geom_audit = read_json(OUT / "snrp/geometry_audit.json")
    overlay_audit = read_json(OUT / "snrp/overlay_audit.json")
    analysis = read_json(OUT / "did/did_analysis.json")
    snrp_analysis = read_json(OUT / "snrp/snrp_analysis.json")
    packed = read_json(OUT / "qa/lossless_alignment_pack.json")
    projection_error = next(item["max_absolute_error"] for item in snrp_qa["checks"] if item["name"] == "svd_630_snr_p_db")
    anomalies = rows(OUT / "snrp/anomalies.csv")
    overlay_anomalies = rows(OUT / "snrp/overlay_anomalies.csv")
    if anomalies or overlay_anomalies:
        raise ValueError("Anomalous rows require adjudication before this release can be published")
    numeric_cells = []
    arm_table = []
    seed_table = []
    interval_statements = []
    for row in did:
        identifier = f"{row['architecture']} {row['snr_db']} dB"
        vals = [number(row[name], 4, 100) for name in ("M_EE", "M_IE", "M_EI", "M_II", "g_E", "g_I", "gamma", "seed_sd")]
        ci = f"[{number(row['patient_ci_low'], 4, 100)}, {number(row['patient_ci_high'], 4, 100)}]"
        arm_table.append([row["architecture"], row["snr_db"], *vals, ci, f"{row['n_valid_draws']}/{row['n_draws']}"])
        seed_values = json.loads(row["seed_values"])
        if not isinstance(seed_values, dict) or set(seed_values) != {str(s) for s in cfg["scope"]["p0"]["train_seeds"]}:
            raise ValueError("Seed values must name all five fixed seeds")
        seed_table.append([row["architecture"], row["snr_db"], *[number(seed_values[str(seed)], 6, 100) for seed in cfg["scope"]["p0"]["train_seeds"]]])
        low, high = float(row["patient_ci_low"]), float(row["patient_ci_high"])
        interval_statements.append(f"{identifier}：条件患者区间{'未含零' if low > 0 or high < 0 else '包含零'}。")
        numeric_cells.append({"source": "did/did_summary.csv", "key": {"architecture": row["architecture"], "snr_db": int(row["snr_db"])}, "values": {name: float(row[name]) for name in ("M_EE", "M_IE", "M_EI", "M_II", "g_E", "g_I", "gamma", "seed_sd", "patient_ci_low", "patient_ci_high")}, "display_multiplier": 100})
    geometry_table = [[row["snr_db"], row["condition"], number(row["mean"]), number(row["median"]), f"[{number(row['p05'])}, {number(row['p95'])}]", row["n_rows"], row["n_records"], row["n_patients"], row["n_noise_bases"], row["n_anomalous"]] for row in snr_rows]
    for row in snr_rows:
        numeric_cells.append({"source": "snrp/geometry_summary.csv", "key": {"snr_db": int(row["snr_db"]), "condition": row["condition"], "metric": "snr_p_db"}, "values": {name: float(row[name]) for name in ("mean", "median", "p05", "p95")}, "display_multiplier": 1})
    reversal_rows = [row for row in did if float(row["g_E"]) * float(row["g_I"]) < 0]
    reversal_zh = " ".join(f"{row['architecture'].upper()} {row['snr_db']}dB 的均值训练排序反转：E测试下 g_E={number(row['g_E'], 4, 100)}pp，I测试下 g_I={number(row['g_I'], 4, 100)}pp。" for row in reversal_rows)
    reversal_en = " ".join(f"For {row['architecture'].upper()} at {row['snr_db']} dB, the mean training contrast reversed from {number(row['g_E'], 4, 100)} pp under E testing to {number(row['g_I'], 4, 100)} pp under I testing." for row in reversal_rows)
    paper_parts = []
    for row in did:
        paper_parts.append(f"{row['architecture'].upper()} at {row['snr_db']} dB: DiD {number(row['gamma'], 4, 100)} pp, seed SD {number(row['seed_sd'], 4, 100)} pp, conditional patient 95% interval [{number(row['patient_ci_low'], 4, 100)}, {number(row['patient_ci_high'], 4, 100)}] pp")
    geometry_zero = {r["condition"]: r for r in snr_rows if int(r["snr_db"]) == 0}
    geometric_values = "; ".join(f"{c}: {number(geometry_zero[c]['mean'], 6)} dB" for c in cfg["scope"]["p1"]["conditions"])
    paper = ("In an exploratory paired difference-in-differences analysis restricted to held-out electrode-support combinations and 15/5 dB Gaussian perturbations, we compared electrode versus independent-RMS augmentation under electrode and independent-RMS testing. "
             + "; ".join(paper_parts) + ". Each result averages five fixed training seeds, ten held-out combinations and five noise bases; the conditional intervals use 2,000 common patient-cluster multiplicity draws over 2,158 ECGs from 1,877 patients. "
             + reversal_en + " "
             + "Separately, direct float64 projection of the reconstructed, hash-verified signed-control inputs yielded the following mean absolute projected SNRs at nominal 0 dB: " + geometric_values + ". "
             + "The geometry and the unchanged prediction-derived AUROC curves are displayed side by side, without exposure adjustment, equivalence claims or causal mediation interpretation.")
    report = [
        "# ICASSP overnight supplement v2: paired protocol DiD and projected-SNR reanalysis", "",
        f"生成时间：{now()}。基线提交 `{cfg['baseline_commit']}`。本报告仅包含通过独立复算与来源保护验收的新数字。", "",
        "## 1. 发布状态与范围", "",
        "- P0：`P0-COMPLETE`。Gaussian、heldout 组合、15/5 dB、ResNet/TCN、五个固定训练 seeds；仅 absolute macro-AUROC。未分析 train 组合。",
        f"- P1：几何层 `{snrp_status['geometry_status']}`；性能层 `{snrp_status['performance_overlay_status']}`。两层分别验收，不将分类预测作为几何量计算的先决条件。",
        "- P2：独立概率/患者权重复算、独立 SVD 投影与来源保护通过。精确覆盖范围见第 5 节，不把抽查称作全体独立复算。",
        "- P3：`DEFERRED — inventory only; no model execution performed.`",
        "- `training_invocations = 0`；`inference_invocations = 0`。没有加载 checkpoint 做前向，没有新预测、阈值或 checkpoint 重选，没有新增 p 值或多重比较家族。", "",
        "## 2. P0：完整四臂与配对差中差", "",
        "第一个下标为训练策略，第二个为测试结构。E=electrode，I=independent-RMS。g_E=M_EE−M_IE，g_I=M_EI−M_II，Γ=g_E−g_I。正 Γ 表示 electrode training 相对优势在 E 测试结构中更大，不等于总体训练优势。", "",
        "四臂 absolute AUROC 以百分数 (%) 展示；g、Γ、seed SD 与区间均为百分点 (pp)。原始 CSV 全部统一为 fraction。点估计来自全体测试患者，不是 bootstrap 均值。", "",
        table(["架构", "SNR dB", "M_EE %", "M_IE %", "M_EI %", "M_II %", "g_E pp", "g_I pp", "Γ pp", "seed SD pp", "条件患者 95% CI pp", "有效/总 draws"], arm_table), "",
        "### 固定训练 seeds 的 Γ", "",
        table(["架构", "SNR dB", "17 pp", "29 pp", "43 pp", "101 pp", "202 pp"], seed_table), "",
        "每个架构/SNR：50 个 combination×noise 单元；先在同 case、同 draw 作训练策略差，再等权聚合十个 heldout 组合与五个 noise bases，形成 seed 内 DiD，最后等权平均五个 seeds。seed SD 使用 ddof=1，与条件患者 CI 分开。共同患者 multiplicity 保留患者全部 ECG；缺类 draw 为 NaN，不重抽、不填零，只纳入五个 seed 均有效的 draw。", "",
        " ".join(interval_statements) + " 这些区间条件于固定 checkpoint、固定噪声和共同患者抽样，不覆盖训练随机性与噪声生成分布；不作新增显著性判断。", "",
        reversal_zh + " TCN15dB的两个训练对比均为负，且其Γ条件区间包含零；因此不能把评价协议依赖概括成所有条件下的训练优势。以上排序仅指五个固定seed的均值。", "",
        "### 四臂身份规则", "",
        "四臂共享 record_id、patient_id、标签、类别顺序、clean 身份、heldout combo、SNR、noise base 与患者抽样映射。同 E 测试结构内比较相同 X_E，同 I 结构内比较相同 X_I；不要求 X_E=X_I。同一训练策略跨测试结构保持 checkpoint 相同，不同训练策略 checkpoint 不同。原 phase2 diagnostics 未存逐记录 hash；本次从 immutable factorized cache 重建相同 normalized float32 输入，核对全数组 SHA，再记录逐记录 SHA。未保存新波形。", "",
        "- [四臂来源索引](../did/four_arm_alignment.parquet)；[逐记录输入身份](../did/record_input_identity.parquet)；[case 明细](../did/did_long.csv)；[完整摘要](../did/did_summary.csv)；[患者 draw 审计](../did/did_draw_audit.parquet)。", "",
        "## 3. P1：absolute projected SNR 几何诊断", "",
        "路线 A：读取原始 immutable clean/base cache，在内存中按既有 float32 sign→scale→add 规则恢复最终输入 z；逐 case 与逐记录核对 SHA。实际噪声 n=float64(z)−float64(x)，用 float64、rcond=1e−12 构造 P=A A†，直接计算 SNR_P=10 log10(||Px||²/||Pn||²)。实际 total SNR 与 q 恒等式仅作交叉核验；没有用 nominal SNR 或报告四舍五入均值代替实际能量。", "",
        "投影分子为零优先输出 null；非零分子、零分母输出 +inf；没有 epsilon、q 截断或异常置零。下面的 5%–95% 为记录/固定噪声暴露的描述性分位范围，**不是患者 CI**。六个固定 noise bases 各含相同记录数，合并分布给予其等权；五个模式分别显示，不视作随机独立重复。", "",
        table(["nominal dB", "结构", "mean SNR_P dB", "median dB", "P05–P95 dB", "暴露行数", "唯一 ECG", "唯一患者", "固定 noises", "异常"], geometry_table), "",
        "![Projected-SNR geometry distributions](../snrp/geometry_distribution.png)", "",
        "- [逐记录绝对 SNR_P 与能量](../snrp/snr_p_per_record.parquet)；[全部几何量摘要](../snrp/geometry_summary.csv)；[按 noise 分层](../snrp/geometry_by_noise.csv)。",
        f"- 几何异常行数：{len(anomalies)}；overlay 异常行数：{len(overlay_anomalies)}。边界定义与合成解析核验见 [boundary_cases.json](../snrp/boundary_cases.json)。", "",
        "## 4. P1：既有性能曲线与类别暴露的并列展示", "",
        "使用既有 E/I/五个 S 模式预测的原始概率复算 AUROC；曲线先等权平均六个固定 noise bases，再等权平均三个训练 checkpoints。各模式单独展示；E/I 不为五个模式复制权重。source_prediction_id 保留原始预测文件 SHA。756 个 noisy 来源各连接 2158 ECG，共 1,631,448 行；六个 clean 来源仅用于独立参考指标，不伪造 clean 几何 case。没有存储新的逐记录概率副本。", "",
        "![Nominal SNR: unchanged AUROC curves above geometric distributions](../snrp/nominal_snr_overlay.png)", "",
        "真实五类分别显示正/负例的几何暴露分布；多标签记录可属于多个正类，未把类别当作互斥分组，也未按模型/seed 重复计入同一几何暴露。", "",
        "![Positive and negative class exposure distributions](../snrp/class_exposure_distribution.png)", "",
        "- [逐来源 AUROC](../snrp/auroc_overlay.csv)；[逐记录连接审计](../snrp/overlay_alignment.parquet)；[类别暴露分层](../snrp/class_exposure_summary.csv)。三个图均提供 PDF/PNG/SVG，版本与哈希见 [figure_manifest.json](../snrp/figure_manifest.json)。", "",
        "### 对原解释的限制", "",
        "相同 nominal total SNR 不代表相同 projected SNR。E/S/I 的性能排序与几何暴露可以并列观察，但本轮没有控制投影暴露后的 AUROC 估计量，不能据此证明独立于子空间暴露差异的方向效应，不能声称因果中介、等效性或真实硬件效应。既有 signed-control 的固定映射、100 Hz PTB-XL、Gaussian、两个网络边界不扩大；新几何量不是完整生理合理性度量。未做分箱标准化、重加权 AUROC、propensity、等暴露曲线或中介分解。", "",
        "## 5. P2 独立复算与异常审计", "",
        f"- P0：从4000原始 NPZ 独立计算1000组四臂 full-cohort point 与 draw0/1999，核验首尾固定 QA units、四个总体组及20个 seed 组。使用独立正负样本加权计数与半权 ties，不调用生产 AUROC/聚合函数。最大绝对误差 `{did_qa['max_absolute_error']}`；比较数 `{did_qa['comparison_count']}`。其余1998draw 检查缺类有效性、NaN、seed平均、差值和 percentile 一致性，未声称逐概率独立重算全部draw。",
        f"- P1：固定记录 `{snrp_analysis['qa_record_ids']}` ×126cases=630条独立重建，独立 SVD 子空间投影；完整271908行几何主键/hash覆盖，1,631,448行overlay连接审计。42个预定noisy来源与6个clean参考独立复算AUROC，九个解析边界case。检查数 `{snrp_qa['n_checks']}`，失败数 `{snrp_qa['n_failures']}`。",
        f"- 来源保护：重新核对 `{protection['historical_local_sources_checked']}` 个实际消费的历史本地来源字节及 SHA，失败 `{len(protection['failures'])}`。不将此声称为对磁盘所有未消费文件的全量哈希。",
        "- [P0 独立证据](../qa/did_recompute.json)；[P1 独立证据](../qa/snrp_recompute.json)；[来源保护](../qa/source_protection.json)；[逐项发布清单](../qa/release_checklist.md)。", "",
        "## 6. 资源、环境与未执行项目", "",
        f"预冻结代表分片：读文件 {benchmark['file_read_seconds']:.6f}s；macro-AUROC {benchmark['macro_auroc_seconds']:.6f}s；一次共同患者 draw {benchmark['one_patient_draw_seconds']:.6f}s；2158记录投影扫描 {benchmark['projection_scan_seconds']:.6f}s。OS缓存状态未受控，不声称冷缓存吞吐。P0四个worker、P1一个worker，BLAS均为1；GPU不参与。", "",
        f"实际数值生产耗时：P0来源/身份可用性判定 {did_status['availability_elapsed_seconds']:.3f}s，2000draw计算 {did_status['bootstrap_elapsed_seconds']:.3f}s；P1纠错后完整生产 {snrp_status['elapsed_seconds']:.3f}s。P0独立复算 {did_qa['elapsed_seconds']:.3f}s。这些是各阶段实测elapsed，不与OS冷缓存或未来GPU训练吞吐等同。", "",
        f"发布时仅调整连接 Parquet 的无损编码布局：{packed['before']['bytes']:,}→{packed['after']['bytes']:,} bytes，{packed['old_row_groups']}→{packed['new_row_groups']} row groups。全部 {packed['rows']:,}×{packed['columns']} 个表格单元、类型、null 和行序经 Arrow 精确相等核验；原始数据和全部数值 CSV 不变，独立 P1 QA 在最终文件上再次通过。见 [lossless_alignment_pack.json](../qa/lossless_alignment_pack.json)。", "",
        f"独立 SVD 复算的 absolute SNR_P 最大绝对误差为 {projection_error:.16g} dB，低于预定1e−10 dB绝对容差；能量项使用预定1e−12相对容差，不混用绝对误差门。逐指标误差见独立 QA 和 release checklist。", "",
        "T+60/T+90仅为数据可用性/资源门，不是科学门；提前发现所有必需预测与 cache，详见 [resource_gates.json](../freeze/resource_gates.json)。运行时间不作为科学失败条件。", "",
        f"Python：`{env['python']}`。包版本：`{json.dumps(env['package_versions'], sort_keys=True)}`。Parquet支持单独安装于本轮 shared/vendor，旧虚拟环境未修改。初始基准脚本使用当前Python不支持的hashlib.file_digest，已在产生基准结果前改为流式SHA-256；原始失败和修复记录保留于 [runtime_setup.json](../freeze/runtime_setup.json)。", "",
        "本轮首次 overlay 审计曾误拒绝54个来源：legacy run_fingerprint 在两个 ResNet 分片间共享，单值字典覆盖了一个分片。已改为 fingerprint×model×owned_case_id 联合匹配并要求唯一所有者，保留原有全部 hash/标签/checkpoint 检查。修复后762来源（其中756 noisy）全部连接通过，独立验收失败为0；初次失败快照和纠错证据保存在 [p1_correction_record.json](../qa/p1_correction_record.json)。这是本轮审计实现错误，不是原始预测/输入身份冲突；失败的 overlay 未获发布，历史工件未修改。", "",
        f"P3仅静态清点：[tsai InceptionTime](https://github.com/timeseriesAI/tsai/blob/{architecture['public_commit']}/tsai/models/InceptionTime.py)，公开commit `{architecture['public_commit']}`、Apache-2.0；未导入模型、forward/backward、GPU smoke或epoch计时。详见 [gate0_status.md](../third_architecture/gate0_status.md)。", "",
        "## 7. 可复现路径与内容身份", "",
        f"- source manifest SHA-256：`{source_info['sha256']}`，文件 [source_manifest.json](../freeze/source_manifest.json)。",
        f"- code manifest SHA-256：`{code_info['sha256']}`，文件 [code_manifest.json](../freeze/code_manifest.json)。",
        "- 全部新增代码与输出位于 `overnight_supplements_v2/`；旧来源只读。原始波形、输入cache、逐记录概率与 isolated vendor 不进入本次Git提交。",
        "- 推送commit SHA属于外层Git发布证据，最终返回单独给出，不在提交自身内容中递归嵌入自身SHA。", "",
        "在仓库根目录、原始本地工件就位后执行（始终设置 PYTHONDONTWRITEBYTECODE=1 与 OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1）：", "",
        "```text",
        "python -B -m pip install --no-deps --target overnight_supplements_v2/shared/vendor pyarrow==21.0.0",
        "python -B -m overnight_supplements_v2.did.pipeline --run",
        "python -B -m overnight_supplements_v2.snrp.run --run",
        "python -B -m overnight_supplements_v2.qa.did_recompute",
        "python -B -m overnight_supplements_v2.qa.snrp_recompute",
        "python -B -m overnight_supplements_v2.freeze.finalize_sources --seal",
        "python -B -m overnight_supplements_v2.freeze.finalize_sources --verify",
        "python -B -m overnight_supplements_v2.reports.build_report",
        "```", "",
        "复跑会更新本轮隔离目录的派生结果，不能覆盖任何历史来源；需要保存本轮发布快照时先另建新的隔离副本。", "",
        "## 8. 论文插入文字（仅已验收数字）", "", paper, "",
    ]
    destination = OUT / "reports/overnight_supplement_final_report.md"
    destination.write_text("\n".join(report), encoding="utf-8")
    manifest = {"status": "rendered_from_independently_accepted_outputs", "rendered_at": now(), "report": file_info(destination), "source_manifest": source_info, "code_manifest": code_info, "numeric_cells": numeric_cells, "paper_insertion": paper, "did_status": did_status, "snrp_status": snrp_status, "geometry_audit": geom_audit, "overlay_audit": overlay_audit}
    write_json(OUT / "reports/manifest.json", manifest)
    print(json.dumps({"status": "rendered", "report": manifest["report"], "numeric_rows": len(numeric_cells), "training_invocations": 0, "inference_invocations": 0}, indent=2))


if __name__ == "__main__":
    main()
