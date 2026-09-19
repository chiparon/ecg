"""Render a static, auditable research report from completed full-cohort results."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import numpy as np
import pandas as pd
from .common import file_info, load_config, read_json, save_json, stage_paths, write_csv
from .plot import BANDS, MODELS, NAMES
from phase2.src.plot_phase2 import SHORT_LABELS


CSS = """
:root{color-scheme:light;--ink:#232323;--muted:#595959;--rule:#d6d6d6;--accent:#0072B2}
*{box-sizing:border-box}html{scroll-behavior:auto}body{margin:0;color:var(--ink);background:white;font:16px/1.75 Georgia,'Noto Serif CJK SC',SimSun,serif}
main{max-width:1120px;margin:56px auto;padding:0 32px}header{border-bottom:2px solid var(--ink);padding-bottom:24px}h1{font-size:32px;line-height:1.35;margin:10px 0 18px}h2{font-size:24px;border-top:1px solid var(--rule);padding-top:28px;margin-top:56px}h3{font-size:19px;margin-top:32px}
p,li{max-width:92ch;text-wrap:pretty}a{color:var(--accent);text-underline-offset:3px}nav{display:flex;flex-wrap:wrap;gap:8px 24px;margin:24px 0;font:14px/1.6 'Microsoft YaHei',sans-serif}
.kicker,.meta,figcaption,summary,thead{font-family:'Microsoft YaHei',sans-serif}.kicker{letter-spacing:.12em;color:var(--muted);font-size:12px}.meta{font-size:13px;color:var(--muted);overflow-wrap:anywhere}
.table-wrap{overflow:auto;margin:20px 0}table{border-collapse:collapse;font-size:13px;line-height:1.5;min-width:700px;width:100%;font-variant-numeric:tabular-nums}th,td{border-bottom:1px solid var(--rule);padding:9px 10px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{font-weight:600;border-top:1px solid var(--ink);background:#f4f4f4}td.label{text-align:left}
figure{margin:28px 0 40px}figure img{width:100%;height:auto;display:block}figcaption{font-size:13px;line-height:1.65;margin-top:12px;color:var(--muted)}.downloads{font-size:12px;white-space:nowrap}details{margin:20px 0}summary{cursor:pointer;font-weight:600;font-size:14px}code,pre{font:12px/1.65 Consolas,monospace}pre{background:#f4f4f4;padding:20px;overflow:auto;border:1px solid var(--rule)}.formula{font-family:Georgia,serif;padding:12px 0;font-size:17px}.note{font-size:14px;color:var(--muted)}footer{border-top:2px solid var(--ink);margin-top:52px;padding-top:20px;font-size:13px}
@media(max-width:760px){main{padding:0 18px;margin:28px auto}h1{font-size:26px}h2{font-size:22px}body{font-size:15px}}
@media print{main{max-width:none;margin:0;padding:0}nav,.downloads{display:none}h2{break-before:page}figure,table{break-inside:avoid}a{color:inherit}details{display:block}}
"""


def _table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(f'<th scope="col">{escape(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def run(config=None):
    cfg = load_config(config)
    paths = stage_paths(cfg, "full")
    verification = read_json(paths["logs"] / "numeric_verification.json")
    plots = read_json(paths["figures"] / "manifest.json")
    if verification["status"] != "passed" or plots["status"] != "completed" or plots["figure_count"] != 15:
        raise ValueError("The report requires completed numerical acceptance and all fifteen real figures")
    if any(item["config_sha256"] != cfg["_config_sha256"] for item in (verification, plots)):
        raise ValueError("Report inputs belong to different frozen protocols")
    paths["reports"].mkdir(parents=True, exist_ok=True)
    source_names = ("new_summary", "phase2_snr_summary", "effect_plane", "pareto", "matrix_structure")
    frames = {name: pd.read_csv(paths["tables"] / f"{name}.csv") for name in source_names}
    diagnostic = pd.read_csv(paths["tables"] / "noise_diagnostics.csv")
    audit_fields = ["effective_cross_lead_correlation", "power_fraction_0_5", "power_fraction_5_15",
                    "power_fraction_15_40", "power_fraction_40_50", "crop_to_long_source_power_ratio",
                    "crop_demeaning_retained_power_fraction"]
    audit = diagnostic[diagnostic.source_id.isin(BANDS)].groupby(["source_id", "condition"], sort=False)[audit_fields].mean().reset_index()
    write_csv(paths["tables"] / "band_noise_audit.csv", audit)
    frames["band_noise_audit"] = audit
    numeric_cells = []

    def number(source, index, column, scale=1, digits=4, signed=False):
        value = float(frames[source].loc[index, column])
        if not np.isfinite(value):
            raise ValueError(f"Undefined report cell: {source}/{index}/{column}")
        text = format(value * scale, f"{'+' if signed else ''}.{digits}f")
        identifier = f"number-{len(numeric_cells) + 1}"
        numeric_cells.append(dict(id=identifier, source=source, row=int(index), column=column,
                                  scale=scale, digits=digits, signed=signed, text=text))
        return f'<span id="{identifier}" data-source="{source}" data-row="{int(index)}" data-column="{column}" data-scale="{scale}">{text}</span>'

    def uncertainty(source, index, prefix=None):
        mean = "estimate" if prefix is None else f"{prefix}_mean"
        sd = "seed_sd" if prefix is None else f"{prefix}_seed_sd"
        low = "patient_ci_low" if prefix is None else f"{prefix}_ci_low"
        high = "patient_ci_high" if prefix is None else f"{prefix}_ci_high"
        return [number(source, index, mean, 100, 3, True), number(source, index, sd, 100, 3),
                '[' + number(source, index, low, 100, 3, True) + ', ' + number(source, index, high, 100, 3, True) + ']']

    figures = {item["id"]: item for item in plots["figures"]}
    def figure(identifier):
        item = figures[identifier]
        base = f"../../figures/full/{identifier}"
        return f'<figure id="{identifier}"><a href="{base}.png"><img src="{base}.png" alt="{escape(item["caption"], quote=True)}"></a><figcaption>{escape(item["caption"])} <span class="downloads"><a href="{base}.pdf">PDF</a> · <a href="{base}.svg">SVG</a> · <a href="{base}.png">PNG</a></span></figcaption></figure>'

    new = frames["new_summary"]
    structure = frames["matrix_structure"]
    matrix_rows = [[escape(str(row.matrix_id)), number("matrix_structure", row.Index, "rank", digits=0),
                    number("matrix_structure", row.Index, "right_nullity", digits=0),
                    number("matrix_structure", row.Index, "left_nullity", digits=0),
                    number("matrix_structure", row.Index, "total_squared_gain", digits=3),
                    number("matrix_structure", row.Index, "mean_abs_offdiagonal_covariance", digits=3)]
                   for row in structure.itertuples()]

    def contrast_table(analysis):
        selected = new[new.analysis.eq(analysis) & new.metric.eq("macro_auroc") & new.outcome.eq("structure_effect")]
        rows = [[MODELS[row.model], escape(NAMES.get(row.source_id, BANDS.get(row.source_id, row.source_id))), str(int(row.snr)),
                 *uncertainty("new_summary", row.Index)] for row in selected.itertuples()]
        return _table(["模型", "矩阵家族 / 频带", "SNR / dB", "E−I 均值 / pp", "训练种子 SD / pp", "条件患者 95% CI / pp"], rows)

    snr1 = new[new.analysis.eq("snr") & new.metric.eq("macro_auroc") & new.outcome.eq("structure_effect")]
    snr_rows = [[MODELS[row.model], str(int(row.snr)), *uncertainty("new_summary", row.Index)] for row in snr1.itertuples()]
    snr2 = frames["phase2_snr_summary"]
    snr2_rows = [[MODELS[row.model], SHORT_LABELS[row.strategy], str(int(row.snr)), *uncertainty("phase2_snr_summary", row.Index)]
                 for row in snr2[snr2.metric.eq("macro_auroc")].itertuples()]
    band_rows = [[BANDS[row.source_id], "E" if row.condition == "electrode" else "I",
                  *[number("band_noise_audit", row.Index, key, 100, 2) for key in ("power_fraction_0_5", "power_fraction_5_15", "power_fraction_15_40", "power_fraction_40_50", "crop_demeaning_retained_power_fraction")]]
                 for row in audit.itertuples()]
    plane = frames["effect_plane"]
    anchors = plane[plane.kind.eq("bandpass") & plane.combo_id.eq("all") & plane.snr.isin([10, 0])]
    plane_rows = [[MODELS[row.model], SHORT_LABELS[row.strategy], str(int(row.snr)),
                   *uncertainty("effect_plane", row.Index, "x"), *uncertainty("effect_plane", row.Index, "y")]
                  for row in anchors.itertuples()]
    pareto = frames["pareto"]
    pareto_rows = [[MODELS[row.model], SHORT_LABELS[row.strategy], *uncertainty("pareto", row.Index, "x"),
                    *uncertainty("pareto", row.Index, "y"), number("pareto", row.Index, "clean_auroc", digits=5),
                    number("pareto", row.Index, "joint_unseen_auroc", digits=5), "是" if row.point_dominated else "否"]
                   for row in pareto.itertuples()]
    electrode_rows = {model: pareto[pareto.model.eq(model) & pareto.strategy.eq("electrode")].index.item() for model in MODELS}
    endpoints = {model: snr1[snr1.model.eq(model) & snr1.snr.eq(0)].index.item() for model in MODELS}
    key_rows = {(row.analysis, row.model, row.source_id, int(row.snr)): row.Index
                for row in new[new.metric.eq("macro_auroc") & new.outcome.eq("structure_effect")].itertuples()}
    worker_reports = [read_json(paths["logs"] / f"inference_{model}_{shard}.json") for model, shard in (("resnet", 0), ("resnet", 1), ("tcn", 0))]
    hardware_rows = [[MODELS[row["model"]], str(row["shard_index"]), escape(row["host"]), escape(row["device_name"]),
                      str(row["observed_cells"]), escape(row["software"]["packages"]["torch"]), escape(row["software"]["packages"]["numpy"])]
                     for row in worker_reports]
    headings = [("scope", "范围与统计口径"), ("matrix", "A 矩阵消融"), ("snr", "SNR×结构"), ("bands", "频带敏感性"),
                ("effects", "评价与训练效应"), ("pareto-section", "清洁代价与鲁棒收益"), ("limits", "结论边界"), ("reproduce", "复现与验收")]
    nav = ''.join(f'<a href="#{identifier}">{title}</a>' for identifier, title in headings)
    links = ''.join(f'<li><a href="../../tables/full/{name}.csv">{name}.csv</a></li>' for name in (
        "new_summary", "new_seed_effects", "new_noise_effects", "random_matrix_effects", "phase2_snr_summary",
        "phase2_snr_seed_effects", "effect_plane", "effect_plane_seed_values", "pareto", "pareto_seed_values",
        "matrix_structure", "matrix_electrodes", "matrix_leads", "matrix_column_similarity", "band_noise_audit", "legacy_bridge"))
    body = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ECG 方法学与信号处理补充实验｜最小五项包</title><style>{CSS}</style></head>
<body><!-- Research supplement: laptop reading and print. Existing scientific palette; real frozen tables only. No external assets, scripts, placeholders or decorative dashboard elements. -->
<main><header><div class="kicker">ECG ROBUSTNESS / METHODOLOGICAL SUPPLEMENT</div><h1>方法学与信号处理补充实验<br>最小五项包 · 完整测试集</h1>
<p>本报告区分两个问题：<strong>噪声结构如何改变评价结果</strong>，以及<strong>针对该结构训练是否带来收益</strong>。第一阶段新增推理覆盖全部指定矩阵、强度与频带；第二阶段复用既有预测和检查点，不重新训练。</p>
<p class="meta">2,158 条 ECG · 1,877 名患者 · 新增执行 325 个条件 × 6 个既有检查点 · 2,000 次共同患者重采样<br>冻结配置 SHA-256：{cfg['_config_sha256']}</p></header><nav aria-label="报告目录">{nav}</nav>
<section aria-label="主要结果"><h3>主要观察 · 点估计</h3><ol>
<li>Standard A 在 0 dB 时，E 相对逐导联 RMS 匹配独立噪声的 AUROC 差值：ResNet {number('new_summary', endpoints['resnet'], 'estimate', 100, 3, True)} pp，TCN {number('new_summary', endpoints['tcn'], 'estimate', 100, 3, True)} pp；保留采集结构的噪声在这一设置下更难。</li>
<li>Precordial-only 的 0 dB 差值靠近零：ResNet {number('new_summary', key_rows[('matrix','resnet','precordial_only',0)], 'estimate', 100, 3, True)} pp、TCN {number('new_summary', key_rows[('matrix','tcn','precordial_only',0)], 'estimate', 100, 3, True)} pp；Limb-only 分别为 {number('new_summary', key_rows[('matrix','resnet','limb_only',0)], 'estimate', 100, 3, True)} pp、{number('new_summary', key_rows[('matrix','tcn','limb_only',0)], 'estimate', 100, 3, True)} pp。近零点估计不构成等效证明。</li>
<li>频带敏感性不同：0 dB 时，ResNet 在 5–15 / 15–40 Hz 的差值为 {number('new_summary', key_rows[('band','resnet','band_5_15',0)], 'estimate', 100, 3, True)} / {number('new_summary', key_rows[('band','resnet','band_15_40',0)], 'estimate', 100, 3, True)} pp；TCN 在 0.05–5 / 15–40 Hz 为 {number('new_summary', key_rows[('band','tcn','band_0p05_5',0)], 'estimate', 100, 3, True)} / {number('new_summary', key_rows[('band','tcn','band_15_40',0)], 'estimate', 100, 3, True)} pp。</li>
</ol><p class="note">这三条均为完整结果中的预设条件点估计；各训练种子 SD、条件患者 CI 及其余全部条件见下文，不转换为新增显著性结论。</p></section>
<section id="scope"><h2>范围与统计口径</h2>
<p>本轮只执行已选定的五项：A 矩阵消融、两阶段五档 SNR 曲线、四频带敏感性、评价效应—训练效应二维图，以及清洁代价—joint-unseen 鲁棒收益 Pareto 图。<strong>不包含 B4、频率依赖协方差匹配、500 Hz、外部数据或新增增强训练。</strong></p>
<ul><li>新增第一阶段推理：ResNet 在 DGX Spark 上两路并发；TCN 在本地 RTX 4070 上单路执行。训练种子固定为 17、29、43；每个非清洁设置有六个固定噪声实现。</li>
<li>第二阶段使用原有两模型、四策略、五个训练种子；Gaussian 全电极曲线包含 20、15、10、5、0 dB。两阶段不合并种子，不进行概率集成。</li>
<li>先在每个检查点内按固定噪声实现取指标均值，再报告训练种子均值及样本 SD（ddof=1）。五个随机矩阵先在种子内平均，另保留逐矩阵结果。</li>
<li>患者区间使用与原第二阶段字节一致的 2,000 个共同患者聚类重采样；每次抽中患者时带入其全部 ECG。对同一 draw 先做配对差或比值，再跨训练种子平均。该 CI 条件于已完成的模型、噪声与矩阵，<strong>不是涵盖训练随机性的总不确定性</strong>。</li>
<li>Macro F1 沿用每个检查点的原清洁验证阈值；ECE 是五类正类概率的 15 等宽 bin 校准误差再取宏平均，不是 top-label ECE。AUROC/F1 的 E−I 为正表示 E 测试值更高；ECE 的 E−I 为正表示校准更差。</li>
<li>所有补充比较均为描述性、探索性结果；不新增 p 值或显著性家族，不改变第二阶段原六个 Holm 检验。下文性能差值以百分点（pp）显示，CSV 保留原始 0–1 尺度。</li></ul>
<p class="formula">评价结构效应：Δ<sub>eval</sub> = M(E) − M(I<sub>RMS</sub>)；退化量：D = M(clean) − M(noisy)。</p>
<p class="note">对 ECE，D 为正意味着校准误差下降，而不是性能退化；本文热图仅展示 AUROC 退化，避免混淆方向。</p></section>
<section id="matrix"><h2>一 · A 矩阵消融</h2>
<p>比较 Standard、Limb-only、Precordial-only，以及五个固定随机矩阵的家族均值。Limb/Precordial 操作掩蔽的是<strong>电极列</strong>，不是导联行；Limb-only 仍可通过 Wilson 中央端影响胸导联。每个具体矩阵都重新构建自己的逐记录、逐导联 RMS 匹配独立对照。</p>
<p>随机矩阵固定采用同能量行组内置换与非零连接符号翻转，保持 Standard 的行/列平方范数及非零个数；允许秩改变。这是受限、平衡的接线负对照，不能代表任意随机矩阵总体。掩蔽家族之间并不保证局部导联能量分布相同。</p>
<p class="note">AAᵀ 是独立单位方差电极源在未做逐记录 SNR 标定时的结构参照；不把它等同于最终注入残差的经验协方差。实际输入相关性另行测量。</p>
{_table(['具体矩阵','秩','右零空间维数','左零空间维数','总平方增益','AAᵀ 非对角绝对均值'], matrix_rows)}
{figure('matrix_heatmaps')}{figure('matrix_bipartite')}{figure('matrix_unit_propagation')}{figure('matrix_singular_values')}
<p>Standard 在 0 dB 的 AUROC 结构效应为：ResNet {number('new_summary', endpoints['resnet'], 'estimate', 100, 3, True)} pp，TCN {number('new_summary', endpoints['tcn'], 'estimate', 100, 3, True)} pp。其余家族与 10 dB 结果全部列出，不按效应方向筛选。</p>
{contrast_table('matrix')}{figure('matrix_structure_effects')}{figure('matrix_correlation_drop')}
<p class="note">相关性散点的横轴是 0 dB 最终注入残差在活跃导联上的平均绝对非对角 Pearson 相关；不能把拓扑、秩、局部能量分配与频谱影响都解释成一个相关系数。</p></section>
<section id="snr"><h2>二 · 两阶段 SNR×结构曲线</h2>
<p>第一阶段以相同既有检查点、同一平台执行全部五档，包含补齐的 15 和 5 dB；旧有 20/10/0 dB 及 clean 共 222 个条件—检查点单元用于数值桥接。第二阶段使用原 Gaussian 全电极设置的五档结果，四种训练策略分开绘制。连线只连接固定条件，不拟合或预设连续、单调的剂量反应。</p>
{figure('snr_macro_auroc')}{figure('snr_macro_f1')}{figure('snr_ece')}
<details><summary>第一阶段 AUROC：全部五档数值</summary>{_table(['模型','SNR / dB','E−I 均值 / pp','训练种子 SD / pp','条件患者 95% CI / pp'], snr_rows)}</details>
<details><summary>第二阶段 AUROC：四策略与全部五档数值</summary>{_table(['模型','训练策略','SNR / dB','E−I 均值 / pp','训练种子 SD / pp','条件患者 95% CI / pp'], snr2_rows)}</details>
<p class="note">两阶段使用不同的固定噪声集合及训练种子数。图中阶段差异不能替代同一第二阶段内的训练策略对照。</p></section>
<section id="bands"><h2>三 · 四频带敏感性</h2>
<p>0.05–5、5–15、15–40、0.5–40 Hz 均由 100 秒高斯源生成，源 FFT 网格为 0.01 Hz，共用白噪声后分别带限，截取中央 10 秒、逐导联去均值，再完成逐记录全 12 导联 SNR 标定。新 0.5–40 Hz 长源参照与旧有 10 秒周期性分支严格区分。</p>
<p><strong>0.05 Hz 的存在由长源频率网格验证，不能由 10 秒片段的 0.1 Hz FFT 网格直接分辨。</strong>截断、去均值和 float32 加法后的实际残差重新计算功率占比与 RMS/SNR；不以目标通带代替实际输入诊断。</p>
{figure('band_drop_heatmap')}{contrast_table('band')}{figure('band_structure_effects')}{figure('band_power_effect')}
<details><summary>最终注入残差的频谱与去均值审计</summary>
<p class="note">以下为 ECG×固定噪声实现等权的输入诊断均值，不是统计 CI。功率百分比排除 DC；区间为 [0,5)、[5,15)、[15,40)、[40,50] Hz。去均值保留比例在 SNR 重新标定之前计算。</p>
{_table(['目标频带','结构','<5 Hz / %','5–15 Hz / %','15–40 Hz / %','40–50 Hz / %','去均值保留功率 / %'], band_rows)}</details></section>
<section id="effects"><h2>四 · 评价效应与训练效应</h2>
<p class="formula">x = M(C,E,c,s) − M(C,I,c,s)；y = M(strategy,E,c,s) − M(C,E,c,s)。</p>
<p>C 是第二阶段 clean-only 模型。横纵轴均使用相同第二阶段的五个配对训练种子，c、s 分别为电极组合与 SNR。主图保留全部 21 个 Gaussian 组合×五档 SNR×两模型×三增强策略，共 630 个点；不把第一阶段三种子结果与第二阶段五种子结果错配。</p>
<p><strong>两轴共享 clean-only electrode 基线，统计上不独立。</strong>本图用于区分评价变化与训练收益，不拟合因果斜率，也不把“评价更难/更容易”等同于“增强更好/更差”。</p>
{figure('evaluation_training_plane')}
<details><summary>全电极、10/0 dB 锚点数值；完整 630 个 Gaussian 点见 CSV</summary>{_table(['模型','训练策略','SNR','x 均值 / pp','x 种子 SD','x 患者 95% CI','y 均值 / pp','y 种子 SD','y 患者 95% CI'], plane_rows)}</details>
<p>NSTDB 的 bw、ma、em 另外形成 54 个点，单独展示为回放敏感性分析；波形形态、时序统计和跨导联结构共同变化，不能当成纯结构隔离证据。</p>{figure('evaluation_training_nstdb')}</section>
<section id="pareto-section"><h2>五 · 清洁代价与 joint-unseen 鲁棒收益</h2>
<p class="formula">x = AUROC(C,clean) − AUROC(strategy,clean)；y = R(strategy) − R(C)，R = AUROC(joint-unseen) / (AUROC(clean) + 10<sup>−12</sup>)。</p>
<p>joint-unseen 沿用原主要终点：10 个 held-out 电极组合×15/5 dB×五个噪声实现、electrode 条件。比值先在各检查点及各患者 draw 内构成。<strong>横轴越小、纵轴越大越好，即左上方向</strong>，不是右上。</p>
<p>Electrode 训练相对 clean-only：ResNet 的清洁代价为 {number('pareto', electrode_rows['resnet'], 'x_mean', 100, 3, True)} pp，retention 收益为 {number('pareto', electrode_rows['resnet'], 'y_mean', 100, 3, True)} pp；TCN 分别为 {number('pareto', electrode_rows['tcn'], 'x_mean', 100, 3, True)} pp 和 {number('pareto', electrode_rows['tcn'], 'y_mean', 100, 3, True)} pp。负清洁代价表示清洁 AUROC 点估计提高。</p>
{figure('pareto')}{_table(['模型','策略','清洁代价 / pp','种子 SD','患者 95% CI','retention 收益 / pp','种子 SD','患者 95% CI','清洁 AUROC','joint-unseen AUROC','点估计被支配'], pareto_rows)}
<p class="note">虚线和支配标记仅按同一模型的点估计定义；不表示统计显著支配、临床非劣或临床可接受性。保留绝对 AUROC，避免只看归一化 retention 而忽略分母变化。</p></section>
<section id="limits"><h2>结论边界与投稿表述</h2><ul>
<li>本轮能支持的是固定线性采集映射、指定高斯带限噪声及既有模型下的结构敏感性证据；不是完整电极—皮肤界面、运动伪影或频率依赖阻抗模型的硬件验证。</li>
<li>RMS 匹配控制每条记录每条导联的噪声能量；它不是逐记录频谱或频率依赖协方差的精确匹配。本轮未执行已排除的相关扩展。</li>
<li>训练种子 SD、固定噪声实现间差异、五个随机矩阵间差异和条件患者 CI 是不同变异来源，不能互相替代或直接相加。逐噪声、逐矩阵表保留这些层次。</li>
<li>同一患者、模型和噪声实现参与多个条件；630 个主图点不是 630 个独立样本。不得把视觉斜率、CI 跨零或点支配转换为新增显著性结论。</li>
<li>当前证据仅覆盖既有 100 Hz、同一官方测试划分和两类网络。未测试更高采样率、外部数据、其他电极图或新训练配比；不做这些范围之外的稳健性承诺。</li>
</ul><p>建议把这五项作为对原结果的机制描述与稳健性边界补充，而不是扩张原主要检验家族。原第二阶段正式结论仍以其预注册终点和原六项 Holm 分析为准。</p></section>
<section id="reproduce"><h2>复现与验收</h2>
{_table(['新增推理模型','分片','实际主机','实际 GPU','保存预测单元','PyTorch','NumPy'], hardware_rows)}
<p>DGX 两个实际 ResNet 进程的运行重叠为 {verification['dgx_overlap_seconds']:.3f} 秒。全部 1,950 份新增执行预测的条件、检查点、队列、原阈值及实际输入哈希均经合并检查。旧条件 222 单元的桥接全部通过；阈值标签最小一致率 {100 * verification['bridge']['minimum_label_agreement']:.4f}%，最大概率绝对差 {verification['bridge']['maximum_probability_difference']:.3g}。</p>
<p>最终残差最大 SNR 绝对误差 {verification['noise']['max_final_residual_snr_error_db']:.3g} dB，最大逐导联 RMS 相对误差 {verification['noise']['max_final_residual_lead_rms_relative_error']:.3g}；覆盖 {verification['noise']['diagnostic_rows']:,} 条噪声诊断记录。另对六个确定位置×六个噪声种子×两结构共 72 个数组核对原生成器，0 dB 数组逐元素相同；这一抽样核对不冒充全数组逐元素基准对照。</p>
<p>独立重新聚合核对 930 行第一阶段摘要、120 行第二阶段曲线、684×2 个效应图轴及 8×2 个 Pareto 轴；第二阶段 AUROC 区间直接对照原共同患者 draw。{verification['protected_files_unchanged']} 个受保护基线/协议/结果文件 SHA-256 保持不变。100 记录烟测仅用于技术验收，不进入本报告的科学结果。</p>
<details><summary>主复现命令与顺序</summary><p>在现有项目根目录运行，沿用已完成的第一/二阶段数据、检查点、依赖和原始预测；阶段输入及源文件必须匹配冻结哈希。最初需先完整执行并验收 smoke，再进行 full；不能跳过该闸门。</p>
<p class="note">Smoke 使用相同的 prepare→noise→deploy/predict→pull→merge→analyse_new→verify 链，把下列相应命令的 <code>--stage full</code> 改为 <code>--stage smoke</code>；跳过 analyse_existing、plot、report 和 publication 检查。只有生成通过状态的 smoke_verification.json 后才启动 full。</p>
<p class="note">本轮本地解释器为 <code>phase1_ecg_robustness/.venv/Scripts/python.exe</code>；DGX 工作目录为 <code>/home/chiparon/ecg_methodology_minimal_five</code>，解释器为 <code>/home/chiparon/ecg_benchmark/20260919_134009/venv/bin/python</code>，SSH 别名为 <code>dgx6</code>。下列 python 指相应环境；远端从该工作目录执行，或将其设为 PYTHONPATH。两条 ResNet 命令需在独立进程同时启动。</p>
<pre>python -m methodology_supplement.prepare --stage full
python -m methodology_supplement.noise --stage full
python -m methodology_supplement.analyse_existing --stage full
python -m methodology_supplement.deploy --stage full --direction push

# 本地 RTX 4070：
python -m methodology_supplement.infer --stage full --model tcn --shard-index 0 --shard-count 1
# DGX Spark：同一工作目录、两个独立进程同时执行：
python -m methodology_supplement.infer --stage full --model resnet --shard-index 0 --shard-count 2
python -m methodology_supplement.infer --stage full --model resnet --shard-index 1 --shard-count 2

# 两端全部完成后，在本地：
python -m methodology_supplement.deploy --stage full --direction pull
python -m methodology_supplement.merge --stage full
python -m methodology_supplement.analyse_new --stage full
python -m methodology_supplement.verify --stage full
python -m methodology_supplement.plot --stage full
python -m methodology_supplement.report
python -m methodology_supplement.verify --stage full --publication</pre>
<p class="note">噪声生成、推理、统计和图表报告计算均设置 PYTHONDONTWRITEBYTECODE=1、OMP_NUM_THREADS=1、OPENBLAS_NUM_THREADS=1、MKL_NUM_THREADS=1；推理使用 CUBLAS_WORKSPACE_CONFIG=:4096:8、四个 PyTorch CPU 线程、FP32、batch 128，关闭 AMP/TF32。具体源指纹和软件版本见各 worker JSON，不能把不同软件栈说成位级相同环境。</p></details>
<details><summary>完整表格与来源</summary><ul>{links}</ul><p><a href="../../../configs/minimal_five.json">冻结配置</a> · <a href="../../logs/full/numeric_verification.json">数值验收</a> · <a href="../../figures/full/manifest.json">图形及来源哈希</a> · <a href="../../logs/full/freeze.json">源文件与基线冻结</a> · <a href="../../test_inputs/full/matrices.json">全部矩阵及 Gram/列相似度</a></p></details></section>
<footer>由已验收表格自动生成；所有结果来自完整测试集。差值是效应量，不是显著性声明。生成时间：{datetime.now(timezone.utc).isoformat()}。</footer></main></body></html>"""
    destination = paths["reports"] / "minimal_five_report.html"
    destination.write_text(body, encoding="utf-8")
    result = dict(status="completed", config_sha256=cfg["_config_sha256"], report=file_info(destination),
                  implementation=file_info(Path(__file__)), numeric_cells=numeric_cells,
                  sources={name: file_info(paths["tables"] / f"{name}.csv") for name in frames},
                  figures_manifest=file_info(paths["figures"] / "manifest.json"),
                  numeric_verification=file_info(paths["logs"] / "numeric_verification.json"))
    save_json(paths["reports"] / "manifest.json", result)
    print(f"Saved full scientific report with {len(numeric_cells)} traceable numeric cells", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
