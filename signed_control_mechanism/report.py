"""Generate matched offline HTML/Markdown reports from accepted full-cohort results."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    check_info, file_info, load_config, read_json, require_freeze,
    resolve_path, save_json, stage_paths,
)


# The existing methodology supplement's academic typography, without a plot import.
CSS = """
:root{color-scheme:light;--ink:#232323;--muted:#595959;--rule:#d6d6d6;--accent:#0072B2}
*{box-sizing:border-box}html{scroll-behavior:auto}body{margin:0;color:var(--ink);background:white;font:16px/1.75 Georgia,'Noto Serif CJK SC',SimSun,serif}
main{max-width:1120px;margin:56px auto;padding:0 32px}header{border-bottom:2px solid var(--ink);padding-bottom:24px}h1{font-size:32px;line-height:1.35;margin:10px 0 18px}h2{font-size:24px;border-top:1px solid var(--rule);padding-top:28px;margin-top:56px}h3{font-size:19px;margin-top:32px}
p,li{max-width:92ch;text-wrap:pretty;overflow-wrap:anywhere}a{color:var(--accent);text-underline-offset:3px}nav{display:flex;flex-wrap:wrap;gap:8px 24px;margin:24px 0;font:14px/1.6 'Microsoft YaHei',sans-serif}
.kicker,.meta,figcaption,summary,thead{font-family:'Microsoft YaHei',sans-serif}.kicker{letter-spacing:.12em;color:var(--muted);font-size:12px}.meta{font-size:13px;color:var(--muted);overflow-wrap:anywhere}
.table-wrap{overflow:auto;margin:20px 0}table{border-collapse:collapse;font-size:13px;line-height:1.5;min-width:700px;width:100%;font-variant-numeric:tabular-nums}th,td{border-bottom:1px solid var(--rule);padding:9px 10px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{font-weight:600;border-top:1px solid var(--ink);background:#f4f4f4}
figure{margin:28px 0 40px}figure img{width:100%;height:auto;display:block}figcaption{font-size:13px;line-height:1.65;margin-top:12px;color:var(--muted)}.downloads{font-size:12px;white-space:nowrap}details{margin:20px 0}summary{cursor:pointer;font-weight:600;font-size:14px}code,pre{font:12px/1.65 Consolas,monospace}pre{background:#f4f4f4;padding:20px;overflow:auto;border:1px solid var(--rule)}.formula{padding:12px 0;font-size:17px}.note{font-size:14px;color:var(--muted)}footer{border-top:2px solid var(--ink);margin-top:52px;padding-top:20px;font-size:13px}
@media(max-width:760px){main{padding:0 18px;margin:28px auto}h1{font-size:26px}h2{font-size:22px}body{font-size:15px}}
@media print{main{max-width:none;margin:0;padding:0}nav,.downloads{display:none}h2{break-before:page}figure{break-inside:avoid}a{color:inherit}details{display:block}}
"""
MODELS = {"resnet": "ResNet", "tcn": "TCN"}
FIGURE_IDS = (
    "signed_control_structure_effects", "signed_control_snr_curves",
    "signed_control_matrices", "subspace_q_distributions", "subspace_geometry",
    "input_validation",
)
SECTIONS = (
    ("question", "一 · 研究问题与预先冻结的比较"),
    ("definitions", "二 · E、S、I 的数学定义"),
    ("selection", "三 · 冻结规则与五个完整符号向量"),
    ("validation", "四 · 矩阵不变量与实际输入验收"),
    ("auroc", "五 · 完整 AUROC 结果"),
    ("secondary", "六 · F1 与 ECE 补充结果"),
    ("subspace", "七 · 子空间 q(v) 诊断"),
    ("modes", "八 · 五个 signed mode 的描述性变化"),
    ("interpretation", "九 · 机制解释与反向、不稳定结果"),
    ("limits", "十 · 局限性与不可外推范围"),
    ("reproduce", "十一 · 复现、来源与验收状态"),
)
EXPECTED_ROWS = {
    "signed_control_summary": 54, "signed_control_seed_effects": 162,
    "signed_control_noise_effects": 972, "signed_mode_effects": 810,
    "signed_mode_summary": 270, "absolute_metrics": 150,
    "matrix_validation": 5, "subspace_diagnostics_summary": 253,
    "reuse_bridge": 222,
}


@dataclass(frozen=True)
class Cell:
    html: str
    text: str


def literal(value):
    text = str(value)
    return Cell(escape(text), text)


class Document:
    """One content stream, with shared formatted cells for both publication formats."""

    def __init__(self, frames, report_dir):
        self.frames = frames
        self.report_dir = report_dir
        self.html = []
        self.markdown = []
        self.numeric_cells = []
        self.markdown_tables = []

    def number(self, source, row, column, scale=1, digits=4, signed=False, notation="f"):
        value = float(self.frames[source].loc[row, column])
        if not np.isfinite(value):
            # Nulls remain visibly null; never turn undefined q or CI into zero.
            return literal("null")
        text = format(value * scale, f"{'+' if signed else ''}.{digits}{notation}")
        identifier = f"number-{len(self.numeric_cells) + 1}"
        self.numeric_cells.append(dict(id=identifier, source=source, row=int(row), column=column,
                                       scale=scale, digits=digits, signed=signed, notation=notation, text=text))
        return Cell(f'<span id="{identifier}" data-source="{escape(source)}" data-row="{int(row)}" '
                    f'data-column="{escape(column)}" data-scale="{scale}">{text}</span>', text)

    def paragraph(self, text, css=""):
        self.html.append(f'<p class="{css}">{escape(text)}</p>')
        self.markdown.append(text + "\n")

    def section(self, identifier):
        title = dict(SECTIONS)[identifier]
        self.html.append(f'<h2 id="{identifier}">{escape(title)}</h2>')
        self.markdown.append(f'<a id="{identifier}"></a>\n\n## {title}\n')

    def table(self, key, headers, rows, details=None):
        if any(entry["key"] == key for entry in self.markdown_tables):
            raise ValueError(f"Duplicate report table key: {key}")
        cells = [[cell if isinstance(cell, Cell) else literal(cell) for cell in row] for row in rows]
        if any(len(row) != len(headers) for row in cells):
            raise ValueError(f"Malformed report table: {key}")
        plain_rows = [[cell.text for cell in row] for row in cells]
        self.markdown_tables.append(dict(key=key, headers=headers, rows=plain_rows))
        html = '<div class="table-wrap"><table><thead><tr>' + ''.join(
            f'<th scope="col">{escape(header)}</th>' for header in headers) + '</tr></thead><tbody>'
        html += ''.join('<tr>' + ''.join(f'<td>{cell.html}</td>' for cell in row) + '</tr>' for row in cells)
        html += '</tbody></table></div>'
        if details:
            html = f'<details><summary>{escape(details)}</summary>{html}</details>'
            self.markdown.append(f"### {details}\n")
        self.html.append(html)
        def md_escape(text):
            return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")
        self.markdown.extend([
            f"<!-- table:{key} -->",
            "| " + " | ".join(md_escape(h) for h in headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(md_escape(c) for c in row) + " |" for row in plain_rows],
            f"<!-- /table:{key} -->\n",
        ])

    def code(self, text, title=None):
        html = f"<pre>{escape(text)}</pre>"
        if title:
            html = f"<details><summary>{escape(title)}</summary>{html}</details>"
            self.markdown.append(f"### {title}\n")
        self.html.append(html)
        self.markdown.append(f"```text\n{text}\n```\n")

    def link(self, path, label):
        relative = Path(os.path.relpath(path, self.report_dir)).as_posix()
        self.html.append(f'<p><a href="{escape(relative, quote=True)}">{escape(label)}</a></p>')
        self.markdown.append(f"[{label}]({relative})\n")

    def figure(self, item):
        files = {suffix: Path(os.path.relpath(check_info(info), self.report_dir)).as_posix()
                 for suffix, info in item["files"].items()}
        caption = item["caption"]
        links = ' · '.join(f'<a href="{escape(files[suffix], quote=True)}">{suffix.upper()}</a>'
                           for suffix in ("png", "pdf", "svg"))
        self.html.append(f'<figure id="{escape(item["id"])}"><a href="{escape(files["png"], quote=True)}">'
                         f'<img loading="lazy" src="{escape(files["png"], quote=True)}" '
                         f'alt="{escape(caption, quote=True)}"></a><figcaption>{escape(caption)} '
                         f'<span class="downloads">{links}</span></figcaption></figure>')
        self.markdown.append(f'<a id="{item["id"]}"></a>\n\n![{caption}]({files["png"]})\n\n' +
                             " · ".join(f'[{suffix.upper()}]({files[suffix]})' for suffix in ("png", "pdf", "svg")) + "\n")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _effects(doc, source, frame, key, include_mode=False, include_mode_sd=True, details=None):
    headers = ["模型", "SNR / dB"] + (["模式"] if include_mode else []) + [
        "对比", "均值 / pp", "训练 seed SD / pp", "患者 CI 下限 / pp", "患者 CI 上限 / pp",
        "noise SD / pp"] + (["mode SD / pp"] if include_mode_sd else []) + ["draw 数", "无效 draw 数"]
    rows = []
    for index, row in frame.iterrows():
        cells = [MODELS[row.model], doc.number(source, index, "snr", digits=0)]
        if include_mode:
            cells.append(row["mode"])
        cells.append(row.contrast)
        cells.extend(doc.number(source, index, field, 100, 3, field in (
            "estimate", "patient_ci_low", "patient_ci_high")) for field in (
            "estimate", "seed_sd", "patient_ci_low", "patient_ci_high", "noise_sd"))
        if include_mode_sd:
            cells.append(doc.number(source, index, "signed_mode_sd", 100, 3))
        cells.extend(doc.number(source, index, field, digits=0) for field in ("n_bootstrap", "n_invalid"))
        rows.append(cells)
    doc.table(key, headers, rows, details)


def run(config=None):
    cfg = load_config(config)
    paths = stage_paths(cfg, "full")
    freeze = require_freeze(cfg, "full")
    verification_path = paths["logs"] / "full_verification.json"
    verification = read_json(verification_path)
    figures_path = paths["figures"] / "manifest.json"
    plots = read_json(figures_path)
    _require(verification.get("status") == "passed" and verification.get("stage") == "full",
             "Full numerical acceptance must pass before reporting")
    _require(plots.get("status") == "completed" and plots.get("stage") == "full" and
             plots.get("figure_count") == 6, "Report requires all six completed full figures")
    for item in (verification, plots):
        _require(item.get("config_sha256") == cfg["_config_sha256"], "Publication protocol mismatch")
    figures = {item["id"]: item for item in plots["figures"]}
    _require(len(plots["figures"]) == 6 and set(figures) == set(FIGURE_IDS), "Six figure identities differ")
    for item in figures.values():
        _require(set(item["files"]) == {"png", "pdf", "svg"}, "Each figure needs PNG, PDF and SVG")
        for info in (*item["files"].values(), *item["sources"]):
            check_info(info)
    _require(freeze["n_records"] == 2158 and freeze["n_patients"] == 1877, "No smoke or partial cohort in reports")
    frames = {name: pd.read_csv(paths["tables"] / f"{name}.csv") for name in EXPECTED_ROWS}
    for name, count in EXPECTED_ROWS.items():
        _require(len(frames[name]) == count, f"Incomplete full report source: {name}")
    summary = frames["signed_control_summary"]
    _require(not summary.duplicated(["model", "snr", "metric", "contrast"]).any(), "Duplicate summary cells")
    for metric in ("macro_auroc", "macro_f1", "ece"):
        observed = set(map(tuple, summary.loc[summary.metric.eq(metric), ["model", "snr", "contrast"]].values))
        expected = {(model, snr, contrast) for model in cfg["models"] for snr in cfg["snrs"] for contrast in cfg["contrasts"]}
        _require(observed == expected, f"Missing predefined conditions: {metric}")
    audit_columns = ["condition", "snr_abs_error_db", "designed_snr_abs_error_db", "max_lead_rms_relative_error",
                     "max_designed_lead_rms_relative_error", "max_periodogram_peak_relative_error",
                     "max_designed_periodogram_peak_relative_error", "max_base_sign_absolute_error",
                     "max_scaled_sign_absolute_error", "all_pass"]
    frames["input_validation"] = pd.read_csv(paths["tables"] / "input_validation.csv", usecols=audit_columns)
    _require(len(frames["input_validation"]) == 126 * freeze["n_records"], "Incomplete per-record input audit")
    controls = read_json(check_info(freeze["sign_controls"]))
    matrices = read_json(check_info(freeze["matrices"]))
    smoke_path = paths["logs"] / "smoke_verification.json"
    smoke = read_json(smoke_path)
    _require(smoke.get("status") in ("passed", "waived_by_user"), "Missing smoke acceptance or explicit user waiver")
    manifest = read_json(paths["inputs"] / "manifest.json")
    _require(manifest.get("status") == "completed" and manifest.get("config_sha256") == cfg["_config_sha256"],
             "Missing accepted full input manifest")
    workers = [read_json(paths["logs"] / f"inference_{model}_{shard}.json")
               for model, shard in (("resnet", 0), ("resnet", 1), ("tcn", 0))]
    for worker in workers:
        _require(worker.get("status") == "completed" and worker.get("stage") == "full" and
                 worker.get("config_sha256") == cfg["_config_sha256"], "Incomplete or stale inference worker")
    paths["reports"].mkdir(parents=True, exist_ok=True)
    doc = Document(frames, paths["reports"])
    title = "保秩保谱 Signed-Control 机制实验 · 完整测试集报告"
    doc.paragraph("探索性机制分析；仅使用完整测试队列结果。两个模型共同作为 0 dB E−S macro-AUROC 主要终点，不根据结果选择主要模型。", "meta")
    doc.section("question")
    doc.paragraph("研究问题：在保持低秩、总能量、逐导联能量与单导联频谱后，Standard 电极—导联映射在原始 12 导联坐标中的特定方向和符号拓扑是否仍额外改变分类性能？本轮不重新训练、微调、选阈值或选模型。")
    doc.paragraph("冻结共同主要比较：ResNet 与 TCN 各自的 0 dB macro-AUROC ΔE−S。预设次要比较为 5/10 dB E−S，以及全部 0/5/10 dB S−I、E−I；F1、ECE、五模式差异与子空间诊断均是描述性次要结果。负 ΔE−S 表示 Standard 的 AUROC 更低。")
    doc.paragraph("PTB-XL 官方患者级测试划分：2,158 条 ECG、1,877 名患者；100 Hz、10 秒，float32 [12,1000]，单位 mV。类别顺序 NORM、MI、STTC、CD、HYP；训练 seeds 17、29、43；固定 noise seeds 8128、101、202、303、404、505。五个 signed modes 等权，不是五个独立训练重复。")
    doc.paragraph("同一患者 draw、checkpoint、noise seed 和 mode 内先计算指标差；每个训练 seed 内等权平均五模式和六噪声，再平均三个训练 seeds。这里是指标平均，不是概率集成。seed SD 为三个 seed 估计的样本 SD（ddof=1）；noise SD 和 mode SD 分别描述固定六噪声、五模式的变化，不能相加成总 CI。")
    doc.paragraph("患者 95% CI 使用字节复用的 2,000 次共同患者簇重采样：抽中患者时包括该患者全部 ECG。缺失类别 AUROC draw 保持 NaN，不重抽；普通均值传播无效性，仅对有效聚合 draw 取 2.5%/97.5% 分位数，并保留无效数。该 CI 条件于已有 checkpoints、固定 noises 与 modes，不覆盖训练、噪声和模式共同变化的总不确定性。")
    doc.section("definitions")
    doc.paragraph("E：nE = Aε；S：Sk = diag(sk)，AS,k = SkA，nS,k = Sk nE；I：沿用冻结的同记录、同导联 RMS 匹配 independent-RMS 噪声。A 是 12×9 理想电极—导联线性映射。", "formula")
    doc.paragraph("XE = Xclean + α nE；XS,k = Xclean + α Sk nE；XI = Xclean + α nI。clean ECG 完全不翻转；S 在原 Standard float32 base-noise 上按导联乘固定符号，然后使用 E 的同一个 SNR 因子；不重新抽电极源，不为 S 单独标定。现有 base 已按记录标定为 0 dB；目标 SNR 的附加缩放为 float32 10^(−SNR/20)。", "formula")
    doc.paragraph("C = AAᵀ，CS,k = Sk C Skᵀ；Sk 为正交对角矩阵，因此秩、奇异值、协方差特征值、对角方差、每导联 RMS、每导联 periodogram 与总能量保持。改变的是非对角协方差符号布局和相对 clean/原始导联坐标的子空间方向。上述精确性质适用于设计噪声；float32 加到 clean 后的实际残差另行审计。")
    doc.paragraph("ΔE−S = M(E) − mean_k M(Sk)；ΔS−I = mean_k M(Sk) − M(I)；ΔE−I = M(E) − M(I)。S 均值是五个模式指标的等权平均。I 只按逐记录、逐导联 RMS 匹配，不声称与 E/S 逐频率的 periodogram 相同。", "formula")
    doc.section("selection")
    doc.paragraph("在任何本轮分类预测或结果筛选之前冻结：PCG64 selection_seed=2026092001；固定第一导联符号 +1，余下 11 位独立均匀抽取 −1/+1；负号数限定 4–7；排除全正和已接受重复；仅接受 ||Sk C Skᵀ−C||F / ||C||F > 10⁻¹⁰，按候选顺序保留最先五个。接受依据与 AUROC/F1/ECE 无关；第一位固定消除全局负号等价形式。")
    doc.paragraph("向量导联顺序：" + ", ".join(controls["lead_order"]) + "。候选编号从 1 开始；五模式是受限平衡的数学方向控制，而不是可推广随机矩阵总体样本。")
    matrix_frame = frames["matrix_validation"]
    control_rows = []
    for index, row in matrix_frame.iterrows():
        vector = json.loads(row.sign_vector)
        control_rows.append([row["mode"], doc.number("matrix_validation", index, "candidate_index", digits=0),
                             "[" + ", ".join(f"{int(sign):+d}" for sign in vector) + "]",
                             doc.number("matrix_validation", index, "negative_count", digits=0),
                             doc.number("matrix_validation", index, "covariance_relative_change", digits=9),
                             row.matrix_array_sha256])
    doc.table("sign_vectors", ["模式", "候选编号", "完整 12 位 sign vector", "负号数", "协方差相对变化", "AS 数组 SHA-256"], control_rows)
    doc.code(json.dumps({"frozen_at": controls["frozen_at"], "generator": controls["generator"],
                         "selection_used_classification_results": controls["selection_used_classification_results"],
                         "acceptance": [{key: control[key] for key in ("mode", "candidate_index", "reason")} for control in controls["controls"]]},
                        ensure_ascii=False, indent=2), "冻结时间、生成器与逐模式接受理由")
    doc.section("validation")
    invariant_fields = ["rank_standard", "rank_signed", "singular_max_abs_error", "eigenvalues_max_abs_error",
                        "covariance_diagonal_max_abs_error", "unit_source_rms_max_abs_error", "offdiagonal_sign_differences"]
    doc.table("matrix_invariants", ["模式", "rank(A)", "rank(SA)", "奇异值最大误差", "特征值最大误差", "对角方差最大误差", "unit-source RMS 最大误差", "非对角符号差异数", "非原矩阵", "通过"],
              [[row["mode"], *[doc.number("matrix_validation", index, field, digits=0 if field.startswith("rank") or field == "offdiagonal_sign_differences" else 16) for field in invariant_fields],
                str(row.different_from_standard), str(row.passed)] for index, row in matrix_frame.iterrows()])
    doc.paragraph("float64 矩阵验收：秩一致；奇异值、协方差特征值、对角方差与 unit-source RMS 最大绝对误差 ≤10⁻¹²；协方差相对变化 >10⁻¹⁰；矩阵不重复且非对角符号确实变化。下方最大输入误差按每个条件遍历全部记录、三个 SNR、六噪声后取最大值，直接引用原始逐记录 CSV 对应行，不把理论不变量替代实测。")
    audit = frames["input_validation"]
    audit_fields = ["snr_abs_error_db", "designed_snr_abs_error_db", "max_lead_rms_relative_error",
                    "max_designed_lead_rms_relative_error", "max_periodogram_peak_relative_error",
                    "max_designed_periodogram_peak_relative_error", "max_base_sign_absolute_error", "max_scaled_sign_absolute_error"]
    audit_rows = []
    for condition in cfg["conditions"]:
        group = audit[audit.condition.eq(condition)]
        cells = [condition]
        for field in audit_fields:
            valid = group[field].dropna()
            cells.append(doc.number("input_validation", valid.idxmax(), field, digits=12) if len(valid) else literal("不适用"))
        audit_rows.append(cells)
    doc.table("actual_input_maxima", ["条件", "实际 SNR 误差/dB", "设计 SNR 误差/dB", "实际 RMS 相对误差", "设计 RMS 相对误差", "实际逐导联谱误差", "设计逐导联谱误差", "base 符号最大误差", "缩放符号最大误差"], audit_rows)
    doc.paragraph("实际残差严格定义为 float64(noisy_float32) − float64(clean_float32)，包含 float32 加法舍入。验收门槛：最终 SNR 误差 ≤10⁻⁴ dB、逐导联 RMS 相对误差 ≤10⁻⁶、E/S periodogram 误差 ≤10⁻⁶。逐导联全频率（含 DC）矩形窗单边 periodogram 的最大绝对 bin 差除以该参考导联峰值，再对导联取最大；不是全导联平均谱，也不是逐 bin 相对误差。零参考谱要求严格为零。I 的谱/符号对比不适用，CSV 用空值和适用标志保留。每个最终输入与每条记录都有 SHA-256；不持久化冗余 noisy/raw 波形。")
    doc.figure(figures["input_validation"])
    doc.figure(figures["signed_control_matrices"])
    doc.code(json.dumps({"standard": matrices["standard"], "projection_checks": {key: value for key, value in matrices["projection"].items() if key not in ("P", "Q")}}, ensure_ascii=False, indent=2),
             "Standard A、AAᵀ、完整谱与投影检查；全部五个矩阵见 matrices.json")
    doc.section("auroc")
    doc.paragraph("下表完整保留 2 模型×3 SNR×3 对比的 18 行，不删除反向、接近零或不稳定结果。差值与 SD/CI 以百分点（pp）报告；源 CSV 是原始 0–1 单位。AUROC 正差值代表第一个条件更好。0 dB E−S 是两个共同主要终点，其余均为事前指定次要比较。")
    _effects(doc, "signed_control_summary", summary[summary.metric.eq("macro_auroc")], "auroc_all")
    doc.figure(figures["signed_control_structure_effects"])
    doc.figure(figures["signed_control_snr_curves"])
    seed_frame = frames["signed_control_seed_effects"]
    primary_seeds = seed_frame[seed_frame.metric.eq("macro_auroc") & seed_frame.snr.eq(0) & seed_frame.contrast.eq("E-S")]
    doc.table("primary_seed_directions", ["模型", "训练 seed", "0 dB E−S / pp"],
              [[MODELS[row.model], doc.number("signed_control_seed_effects", index, "seed", digits=0),
                doc.number("signed_control_seed_effects", index, "estimate", 100, 3, True)] for index, row in primary_seeds.iterrows()])
    doc.section("secondary")
    doc.paragraph("方向警告：AUROC/F1 的正差值表示第一个条件更好；ECE 的正差值表示第一个条件校准更差，不能沿用 AUROC 的优劣方向。F1 沿用每个 checkpoint 的原 clean-validation 阈值（比较为 ≥，zero_division=0）；ECE 为五类正类概率各自 15 个等宽 bin 的 ECE 再作宏平均，不是 top-label ECE。")
    _effects(doc, "signed_control_summary", summary[summary.metric.eq("macro_f1")], "f1_all", details="macro-F1：完整 18 行")
    _effects(doc, "signed_control_summary", summary[summary.metric.eq("ece")], "ece_all", details="macro classwise ECE：完整 18 行；正差表示更差")
    doc.section("subspace")
    doc.paragraph("P = AA†，Q = I12−P；q(v) = ||Qv||F² / ||v||F²。float64 SVD/pseudoinverse 使用 rcond=10⁻¹²；P 对称性与幂等误差要求 <10⁻¹⁰。分母为零时 q=null 并计数，不加 ε、不填零。", "formula")
    doc.paragraph("q(clean) 每记录只计算一次；E_noise/S_noise/I_noise 使用实际 float32 加法后的 float64 残差，E_input/S_input/I_input 使用完整 noisy 输入。所有 q 都相对 Standard 的 P，不使用各模式自己的投影。因此 S_noise 的非零 q 是方向控制设计的一部分，不是实现错误；E_noise 的有限浮点残差不必严格为零。")
    doc.paragraph("下表按 object×SNR×noise seed×mode 分组，每组 mean 是有效记录 q 的算术平均，SD 为 ddof=1，中位数和 Q1/Q3 为线性分位数，IQR=Q3−Q1；null 排除于统计但明确计数。若另行汇总对象的总体描述均值，须先在相同 SNR 下等权平均六个 noise seed 的组均值，S 再等权平均五个 mode；clean 只计一次。该均值聚合不是对合并波形求 q，也不是患者加权，更不能通过平均各组中位数/IQR 得到合并分布的中位数/IQR。这里保留全部 253 组，避免模糊聚合权重。")
    doc.paragraph("q 分布图另以每记录为单位：E/I 及每个 S mode 先平均六个固定 noise realizations，S 总均值再平均五个固定 modes；随后在 ECG 记录间绘制描述性分布。箱图不是患者 CI，也不把各噪声重复当独立患者；遇到 null 不填零，图形的实际有效数与处理口径见图注及原始分组/逐记录 CSV。")
    q_frame = frames["subspace_diagnostics_summary"]
    q_rows = []
    for index, row in q_frame.iterrows():
        q_rows.append([row.object, doc.number("subspace_diagnostics_summary", index, "snr", digits=0) if pd.notna(row.snr) else "—",
                       doc.number("subspace_diagnostics_summary", index, "noise_seed", digits=0) if pd.notna(row.noise_seed) else "—",
                       row["mode"] if pd.notna(row["mode"]) else "—",
                       *[doc.number("subspace_diagnostics_summary", index, column, digits=4, notation="e") for column in ("mean", "sd", "median", "q1", "q3", "iqr", "min", "max")],
                       *[doc.number("subspace_diagnostics_summary", index, column, digits=0) for column in ("n_valid", "n_null")]])
    doc.table("q_grouped_summary", ["对象", "SNR/dB", "noise seed", "mode", "mean", "SD", "median", "Q1", "Q3", "IQR", "min", "max", "有效记录", "null记录"], q_rows,
              details="全部对象、SNR、noise 与 mode 的分组 q 摘要")
    doc.paragraph("Standard rank=8 的列空间包含六个不受约束的胸导联坐标，以及理想肢体导联关系。q 仅诊断这些理想线性约束，低 q 不是完整生理合理性、临床真实性或分类质量证明，也不是临床终点。")
    doc.paragraph("shift_noise 本轮不新增生成。仓库已有 strict_control.py 以及 marginal_shift 预测、offset、seed 和诊断记录，但没有可直接复用的 shifted/noisy 波形数组或最终输入哈希。因此不能严格复用其输入；本轮没有重构一个声称等价的 shift 分支，也没有伪造 shift q。")
    doc.figure(figures["subspace_q_distributions"])
    doc.figure(figures["subspace_geometry"])
    doc.section("modes")
    doc.paragraph("下表保留全部五模式在两个模型 0 dB 下的 AUROC 三对比。逐模式先平均六 noise seeds，再跨三个训练 seeds；训练 seed SD 与条件患者 CI 仍分开。E−I 与 mode 无关，五次重复在原表明确标记 redundant_contrast=true，不得当作五份独立证据。全 SNR、全指标的 270 行 mode summary 与 810 行 seed/mode 表均随完整 CSV 链接提供。")
    mode_frame = frames["signed_mode_summary"]
    _effects(doc, "signed_mode_summary", mode_frame[mode_frame.metric.eq("macro_auroc") & mode_frame.snr.eq(0)],
             "mode_0db_auroc", include_mode=True, include_mode_sd=False)
    doc.section("interpretation")
    doc.paragraph("描述性方向规则对两个共同主要模型分别应用：只有全部三个训练 seed 的 E−S 点估计均 <0 且条件患者 CI 上限 <0，才称该模型具有稳定负向一致性。CI 与 seed SD 不是同一不确定性来源；此规则不是新增假设检验，不生成 p 值或显著性家族。")
    decisions = {}
    decision_rows = []
    for model in cfg["models"]:
        endpoint = summary[summary.model.eq(model) & summary.metric.eq("macro_auroc") & summary.snr.eq(0) & summary.contrast.eq("E-S")]
        index = endpoint.index.item()
        row = endpoint.iloc[0]
        seeds = primary_seeds[primary_seeds.model.eq(model)]
        _require(set(seeds.seed.astype(int)) == set(cfg["phase1_training_seeds"]), "Incomplete co-primary seed directions")
        values = seeds.estimate.to_numpy()
        stable = bool(np.all(values < 0) and row.patient_ci_high < 0)
        reverse = bool(np.all(values > 0) and row.patient_ci_low > 0)
        decisions[model] = "stable_negative" if stable else "stable_reverse" if reverse else "mixed_or_uncertain"
        interpretation = ("三个 seed 均负、CI 上限负：与 Standard 方向额外降低 AUROC 一致。" if stable else
                          "三个 seed 均正、CI 下限正：稳定反向，signed-control 更难；不支持 Standard 更具破坏性的方向假设。" if reverse else
                          "方向或区间不稳定/接近零：该模型没有满足预定稳定负向规则；不构成等效证据。")
        decision_rows.append([MODELS[model], doc.number("signed_control_summary", index, "estimate", 100, 3, True),
                              doc.number("signed_control_summary", index, "seed_sd", 100, 3),
                              doc.number("signed_control_summary", index, "patient_ci_low", 100, 3, True),
                              doc.number("signed_control_summary", index, "patient_ci_high", 100, 3, True), interpretation])
    doc.table("coprimary_interpretation", ["共同主要模型", "0 dB E−S/pp", "seed SD/pp", "CI 下限/pp", "CI 上限/pp", "分别解释"], decision_rows)
    if all(value == "stable_negative" for value in decisions.values()):
        conclusion = "两个共同主要模型均达到稳定负向规则。在本研究固定 A、100 Hz、合成噪声与两种网络条件下，结果与采集映射方向在一般低秩性质之外具有额外作用的解释一致；不能外推为真实设备验证或唯一因果机制。"
    elif any(value == "stable_negative" for value in decisions.values()):
        conclusion = "两个共同主要模型的证据不一致：仅部分模型达到稳定负向规则。额外映射方向支持有限且具有模型依赖性；不得事后把支持方向的模型改称唯一主要模型。其他模型的反向或不稳定结果必须同等保留。"
    else:
        conclusion = "两个共同主要模型均未达到 Standard 更难的稳定负向规则；本轮不支持该额外方向假设。接近零或不稳定结果不能证明 E 与 S 等效，也不能单独证明通用低秩结构是唯一原因；若有反向效应，应直接承认 signed-control 可能更难，而不强制归入“二者相近”。"
    doc.paragraph(conclusion)
    doc.paragraph("两种原先预想的解释均可形成有用机制证据，但不是强制二选一。S−I 和 E−I 提供固定独立对照下的背景，不能将 signed-control 的方向差异、低秩性质、临床真实性或因果归因混为一谈；所有 5/10 dB 与模式差异都保持次要地位。")
    doc.section("limits")
    for text in (
        "A 是理想线性电极—导联关系，不包含完整电极—皮肤界面、右腿驱动、共模抑制、运动伪影或频率依赖阻抗。signed-control 是保秩、保能量、保单导联谱的数学方向控制，不是物理可实现的真实采集错误模型。",
        "范围仅为 PTB-XL 已有 100 Hz 官方切分、ResNet/TCN、三个既有训练 seeds、固定高斯电极源与 0/5/10 dB。五个固定模式不是可推广的随机方向总体，六个噪声不是新增训练重复。",
        "患者 CI 条件于现有模型、固定噪声和固定 modes；不覆盖训练/噪声/方向的总体不确定性。多个条件共享患者、模型和原噪声，不是独立样本；不用跨零、点估计接近或方向规则作等效、临床非劣或唯一因果证明。",
        "不同设备及软件版本不能称为位级相同环境；本轮 E/I 使用已验证的不可变旧预测，没有重新进行 E/I 的跨设备推理。fresh clean 数值检查不等于 E/I 或跨设备数值桥接。",
        "本轮不改变第一/二阶段原预设训练比较或原 Holm 结论；不新增训练、模型选择、阈值搜索、p 值或显著性家族。",
    ):
        doc.paragraph(text)
    doc.section("reproduce")
    doc.paragraph("官方数据来源：PTB-XL 1.0.3，https://physionet.org/content/ptb-xl/1.0.3/ 。只读取本地合法的既有处理结果；本报告和签名控制目录不复制、不上传原始波形。处理身份由原冻结的 metadata_labels_provenance_sha256 及 clean/cohort/draws 文件与数组指纹保留，不把来源网址当作处理内容哈希。")
    doc.code(json.dumps(freeze["data_identity"], ensure_ascii=False, indent=2), "原 PTB-XL 处理身份")
    doc.paragraph(f"基线代码提交：{cfg['base_commit']}。仓库：{cfg['repository']}。该提交标识既有基础，不代表本轮新增且未提交的 signed-control 实现属于该提交。下方列出当前新源文件 SHA-256，并在报告 manifest 中保存实际实现指纹。")
    doc.paragraph(f"冻结配置规范化内容 SHA-256：{cfg['_config_sha256']}；配置文件字节 SHA-256：{file_info(cfg['_config_path'])['sha256']}。二者计算对象不同，不能互换。")
    if smoke["status"] == "waived_by_user":
        doc.paragraph("Smoke 状态：waived_by_user，不是 passed。100 条记录的输入审计和三路 GPU smoke 推理已完成；用户指示“直接开干，别烟测浪费时间”，因此跳过剩余 smoke merge、bootstrap 与 verification，直接进入 full。这是执行门控的明确授权偏离，不是伪造烟测通过；科学配置、五个 sign vectors 与共同主要终点不变。完整授权与已完成部分以 smoke 日志为准。")
    else:
        doc.paragraph("Smoke 状态：passed；smoke 仅作技术验收，其记录和统计不进入本报告科学结果。")
    doc.paragraph("Full 状态：passed（本报告生成前读取 full_verification.json）。本报告生成器完成不代表 publication verification 已通过；出版级数字、Markdown、图像和本地链接验收在生成后由独立步骤执行，其最终记录单独保存在 logs/full/publication_verification.json。为避免报告哈希循环依赖，本报告不把旧 publication 记录冒充当前报告验收，也不将该记录作为生成输入。")
    doc.code(json.dumps(smoke, ensure_ascii=False, indent=2), "实际 smoke 状态与用户授权记录")
    doc.code(json.dumps(verification, ensure_ascii=False, indent=2), "Full 独立数值验收记录")
    doc.paragraph("E/I 的 216 个预测单元及 6 个 clean 单元复用不可变原文件：逐项核对 input/cohort/checkpoint/threshold/model-source/probability SHA，重算原有完整指标。reuse_bridge.csv 中 E/I 的 fresh 概率差和标签一致率为空，表示不适用；没有把文件与自身比较写成概率差为零的数值桥接。另有 6 个 fresh clean 与同分配主机旧 clean 的独立数值比较，列于下表；它不是跨设备桥接，也不计作新增噪声条件。")
    bridge = frames["reuse_bridge"]
    clean_bridge = bridge[bridge.condition.eq("clean")]
    bridge_columns = ("probabilities_max_abs", "fixed_threshold_label_agreement", "fresh_macro_auroc_difference", "fresh_macro_f1_difference", "fresh_ece_difference")
    doc.table("fresh_clean_checks", ["模型", "seed", "最大概率绝对差", "阈值标签一致率", "AUROC 新−旧", "F1 新−旧", "ECE 新−旧"],
              [[MODELS[row.model], doc.number("reuse_bridge", index, "seed", digits=0),
                *[doc.number("reuse_bridge", index, field, digits=9, signed=field.startswith("fresh_")) for field in bridge_columns]]
               for index, row in clean_bridge.iterrows()])
    doc.paragraph("完整网格：126 个 noisy 条件×6 checkpoints=756 noisy 预测；其中新 signed 为 5×3×6×6=540 单元，E/I 复用 216；加 clean 复用 6 共 762 个索引单元。worker 各自拥有 ledger/report，merge 检查无重复、无缺失。每个模型各有 270 个新增 signed 单元；clean 数值检查不重复计入条件。")
    environment = [{key: worker[key] for key in ("model", "shard_index", "host", "device_name", "software", "fp32_settings", "started_at", "finished_at", "elapsed_seconds", "observed_cells")}
                   for worker in workers]
    doc.code(json.dumps(environment, ensure_ascii=False, indent=2), "实际三个 full workers：硬件、精确软件版本、FP32 设置和耗时")
    doc.paragraph("ResNet 分配到 DGX Spark 的两个独立并发进程；TCN 分配到本地 RTX 4070 Laptop GPU。推理 deterministic FP32、batch=128、PyTorch CPU threads=4，关闭 AMP 与 TF32；具体 CUDA/cuDNN、包版本、主机和设置以实际 worker JSON 为准，不将两端不同软件栈写成位级相同。")
    timing_sources = {}
    estimate_path = resolve_path(cfg["results_dir"]) / "logs" / "compute_estimate.json"
    if estimate_path.is_file():
        timing_sources["compute_estimate"] = read_json(estimate_path)
    for name in ("input_generation", "statistics", "evaluation_merge"):
        log_path = paths["logs"] / f"{name}.json"
        if log_path.is_file():
            record = read_json(log_path)
            timing_sources[name] = {key: record[key] for key in ("status", "started_at", "finished_at", "elapsed_seconds", "timings", "distribution_wall_seconds", "cache_compute_seconds", "dgx_overlap_seconds") if key in record}
    doc.paragraph("下面将事前估计与实际日志分开列示。线性估计来自上一轮同队列/模型/FP32 运行，不是本轮观测；worker 耗时见上表，实际 input/statistics/merge 记录见下。不同进程有重叠，不能直接把 worker 秒数求和当端到端墙钟；没有对应时间戳的阶段不编造耗时，也不把实现、排障和报告编写时间混入纯计算估计。")
    doc.code(json.dumps(timing_sources, ensure_ascii=False, indent=2), "事前运行估计与实际阶段时间记录")
    commands = """# Project root; use the existing environment, never retrain.
# Local Python: phase1_ecg_robustness/.venv/Scripts/python.exe
# Remote cwd: /home/chiparon/ecg_methodology_minimal_five
# Remote Python: /home/chiparon/ecg_benchmark/20260919_134009/venv/bin/python
# Each shell: PYTHONDONTWRITEBYTECODE=1, OMP_NUM_THREADS=1,
# OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1, CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m signed_control_mechanism.prepare --freeze-controls-only
# For a fresh conventional smoke: prepare -> inputs -> deploy/infer -> pull ->
# merge -> analyse -> verify with --stage smoke. This run's remaining smoke
# stages were waived by explicit user instruction, not marked passed.
# Full still requires the preserved smoke passed/waived_by_user provenance.
python -m signed_control_mechanism.prepare --stage full
python -m signed_control_mechanism.inputs --stage full
python -m signed_control_mechanism.deploy --stage full --direction push
# Local:
python -m signed_control_mechanism.infer --stage full --model tcn --shard-index 0 --shard-count 1
# DGX: launch the following in two separate concurrent processes:
python -m signed_control_mechanism.infer --stage full --model resnet --shard-index 0 --shard-count 2
python -m signed_control_mechanism.infer --stage full --model resnet --shard-index 1 --shard-count 2
# Local after all workers finish:
python -m signed_control_mechanism.deploy --stage full --direction pull
python -m signed_control_mechanism.merge --stage full
python -m signed_control_mechanism.analyse --stage full
python -m signed_control_mechanism.verify --stage full
python -m signed_control_mechanism.plot --stage full
python -m signed_control_mechanism.report
python -m signed_control_mechanism.verify --stage full --publication"""
    doc.code(commands, "完整复现命令与先后顺序")
    implementation = [file_info(path) for path in sorted(Path(__file__).parent.glob("*.py"))]
    doc.code(json.dumps(implementation, ensure_ascii=False, indent=2), "本轮实际实现源文件指纹（不冒充基线提交内容）")
    doc.code(json.dumps({key: freeze[key] for key in ("clean", "cohort", "draws", "array_hashes", "checkpoint_entries", "sign_controls", "matrices")},
                        ensure_ascii=False, indent=2), "处理数组、原始阈值/checkpoint、draws 与控制矩阵的冻结哈希")
    doc.paragraph("完整 CSV 下载包括所有预设条件、模式、seed/noise 层次、绝对指标、最终输入诊断与预测/复用索引；大体积逐记录 q 表不是原始 ECG 波形。HTML 无外部脚本、字体或图片依赖，使用原生锚点/details 和可横向滚动表格；Markdown 与 HTML 由相同格式化单元生成。")
    all_tables = sorted(paths["tables"].glob("*.csv"))
    for path in all_tables:
        doc.link(path, path.name)
    linked_sources = {
        "frozen_config": Path(cfg["_config_path"]), "freeze": paths["logs"] / "freeze.json",
        "full_verification": verification_path, "smoke_verification": smoke_path,
        "figures_manifest": figures_path, "input_manifest": paths["inputs"] / "manifest.json",
        "sign_controls": check_info(freeze["sign_controls"]), "matrices": check_info(freeze["matrices"]),
        "patient_draws_provenance": paths["inputs"] / "patient_draws_provenance.json",
    }
    for name, path in linked_sources.items():
        doc.link(path, name + " · " + path.name)
    for model, shard in (("resnet", 0), ("resnet", 1), ("tcn", 0)):
        path = paths["logs"] / f"inference_{model}_{shard}.json"
        linked_sources[f"inference_{model}_{shard}"] = path
        doc.link(path, path.name)
    for name in ("input_generation", "statistics", "evaluation_merge"):
        path = paths["logs"] / f"{name}.json"
        if path.is_file():
            linked_sources[name] = path
            doc.link(path, path.name)
    if estimate_path.is_file():
        linked_sources["compute_estimate"] = estimate_path
        doc.link(estimate_path, estimate_path.name)
    generated_at = datetime.now(timezone.utc).isoformat()
    nav = ''.join(f'<a href="#{identifier}">{escape(section)}</a>' for identifier, section in SECTIONS)
    html = (f'<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title>'
            f'<style>{CSS}</style></head><body><!-- Scientific publication: existing academic style; '
            'only accepted full results; native offline navigation; no external assets. -->'
            f'<main><header><div class="kicker">ECG ROBUSTNESS / SIGNED-CONTROL MECHANISM</div><h1>{escape(title)}</h1>'
            f'<p class="meta">冻结配置 SHA-256：{cfg["_config_sha256"]}</p></header>'
            f'<nav aria-label="报告目录">{nav}</nav>' + '\n'.join(doc.html) +
            f'<footer>结果单位及不确定性按冻结协议分列；无新增显著性声明。生成时间：{generated_at}。</footer></main></body></html>')
    markdown = f"# {title}\n\n冻结配置 SHA-256：`{cfg['_config_sha256']}`\n\n" + '\n'.join(
        f"- [{section}](#{identifier})" for identifier, section in SECTIONS) + '\n\n' + '\n'.join(doc.markdown)
    markdown += f"\n生成时间：{generated_at}。出版验收独立于报告生成，最终状态以当前 publication_verification.json 为准。\n"
    html_path = paths["reports"] / "signed_control_final_report.html"
    markdown_path = paths["reports"] / "signed_control_final_report.md"
    html_path.write_text(html, encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    sources = {path.stem: file_info(path) for path in all_tables}
    sources.update({name: file_info(path) for name, path in linked_sources.items()})
    result = dict(status="completed", stage="full", config_sha256=cfg["_config_sha256"],
                  base_commit=cfg["base_commit"], generated_at=generated_at,
                  html=file_info(html_path), markdown=file_info(markdown_path), sources=sources,
                  numeric_cells=doc.numeric_cells, markdown_tables=doc.markdown_tables,
                  sections=[dict(id=identifier, title=section) for identifier, section in SECTIONS],
                  implementation=file_info(Path(__file__)), implementation_fingerprints=implementation,
                  figures_manifest=file_info(figures_path), full_verification=file_info(verification_path),
                  smoke_status=smoke["status"], publication_status="separate_post_generation_verification",
                  interpretation=decisions, conclusion=conclusion)
    save_json(paths["reports"] / "manifest.json", result)
    print(f"Saved matched full HTML/Markdown reports with {len(doc.numeric_cells)} traceable numbers", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
