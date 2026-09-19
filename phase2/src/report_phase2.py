"""Generate the requested Chinese report from verified full-stage measurements."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .common import (
    file_info,
    load_config,
    require_preregistration,
    save_json,
    stage_paths,
)
from .verify_phase2 import audit_figures

CORE = [
    "macro_auroc",
    "macro_ap",
    "macro_f1",
    "micro_auroc",
    "micro_ap",
    "micro_f1",
    "brier",
    "ece",
    "macro_sensitivity",
    "macro_specificity",
    "macro_ppv",
    "macro_npv",
    "clean_agreement",
    "probability_shift",
]
LABELS = {
    "clean_only": "clean-only",
    "independent_rms": "independent-RMS",
    "electrode": "electrode",
    "mixed": "mixed",
}


def number(value, places=4):
    return (
        f"{float(value):.{places}f}"
        if pd.notna(value) and np.isfinite(value)
        else "未定义"
    )


def bounds(low, high):
    return f"[{number(low)}, {number(high)}]"


def probability(value):
    return f"{float(value):.6g}" if pd.notna(value) else "未定义"


def markdown_table(columns, rows):
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
            *[
                "| " + " | ".join(str(value).replace("|", "/") for value in row) + " |"
                for row in rows
            ],
        ]
    )


def _selected_csv(path, predicate):
    chunks = []
    for chunk in pd.read_csv(path, chunksize=100000):
        selected = chunk.loc[predicate(chunk)]
        if len(selected):
            chunks.append(selected)
    if not chunks:
        raise ValueError(f"No requested measurements in {path}")
    return pd.concat(chunks, ignore_index=True)


def build_tables(cfg, stage):
    paths = stage_paths(cfg, stage)
    classes = [
        f"{metric}_{name}"
        for name in cfg["class_order"]
        for metric in ("auroc", "ap", "f1")
    ]
    summary = _selected_csv(
        paths["tables"] / "seed_summary.csv",
        lambda frame: frame.group_id.isin(["clean", "primary_joint"])
        & frame.metric.isin(CORE + classes),
    )
    raw = _selected_csv(
        paths["tables"] / "group_seed_metrics.csv",
        lambda frame: ~frame.group_id.str.contains("__combo_")
        & frame.metric.isin(["macro_auroc", "macro_ap"]),
    )
    indexed = raw.set_index(
        ["model", "strategy", "group_id", "metric", "outcome", "seed"]
    ).value.sort_index()
    seeds = cfg["stages"][stage]["seeds"]
    snrs = cfg["stages"][stage]["test_snrs"]

    def vector(model, strategy, group, metric, outcome="absolute"):
        return (
            indexed.loc[(model, strategy, group, metric, outcome)]
            .reindex(seeds)
            .to_numpy(dtype=float)
        )

    def contrast(model, lhs, rhs, group, **descriptors):
        differences = vector(model, lhs, group, "macro_auroc") - vector(
            model, rhs, group, "macro_auroc"
        )
        ap = vector(model, lhs, group, "macro_ap") - vector(
            model, rhs, group, "macro_ap"
        )
        retention = vector(model, lhs, group, "macro_auroc", "retention") - vector(
            model, rhs, group, "macro_auroc", "retention"
        )
        sd = float(differences.std(ddof=1)) if len(seeds) > 1 else np.nan
        margin = (
            float(stats.t.ppf(0.975, len(seeds) - 1) * sd / np.sqrt(len(seeds)))
            if len(seeds) > 1
            else np.nan
        )
        return dict(
            model=model,
            lhs=lhs,
            rhs=rhs,
            group_id=group,
            **descriptors,
            auroc_difference=float(differences.mean()),
            auroc_difference_sd=sd,
            seed_t95_low=float(differences.mean() - margin),
            seed_t95_high=float(differences.mean() + margin),
            ap_difference=float(ap.mean()),
            retention_difference=float(retention.mean()),
            positive_seed_count=int((differences > 0).sum()),
            n_seeds=len(seeds),
            paired_seed_differences=json.dumps(differences.tolist()),
            inference="secondary descriptive; no added hypothesis test",
        )

    secondary, nstdb, mechanism = [], [], []
    for model in cfg["models"]:
        for condition in cfg["test"]["conditions"]:
            for snr in [value for value in (15, 5) if value in snrs]:
                mechanism.append(
                    contrast(
                        model,
                        "electrode",
                        "independent_rms",
                        f"bandpass__heldout__{condition}__s{snr}",
                        condition=condition,
                        snr=snr,
                    )
                )
        for question, combo_set, strengths in [
            ("unseen_strength", "train", [15, 5]),
            ("unseen_combination", "heldout", [20, 10]),
        ]:
            for snr in [value for value in strengths if value in snrs]:
                group = f"bandpass__{combo_set}__electrode__s{snr}"
                for rhs in ["clean_only", "independent_rms"]:
                    secondary.append(
                        contrast(
                            model, "electrode", rhs, group, question=question, snr=snr
                        )
                    )
        if cfg["stages"][stage]["nstdb"]:
            for kind in cfg["test"]["nstdb_kinds"]:
                for snr in cfg["test"]["nstdb_snrs"]:
                    for rhs in ["clean_only", "independent_rms"]:
                        nstdb.append(
                            contrast(
                                model,
                                "electrode",
                                rhs,
                                f"{kind}__all__electrode__s{snr}",
                                kind=kind,
                                snr=snr,
                                pressure=snr == 0,
                            )
                        )
    return dict(
        summary=summary,
        secondary=pd.DataFrame(secondary),
        nstdb=pd.DataFrame(nstdb),
        mechanism=pd.DataFrame(mechanism),
        primary=pd.read_csv(paths["tables"] / "primary_comparisons.csv"),
        effects=_selected_csv(
            paths["tables"] / "paired_effects.csv",
            lambda frame: frame.group_id.eq("primary_joint"),
        ),
        noise=pd.read_csv(paths["tables"] / "noise_contrasts.csv"),
        ranks=_selected_csv(
            paths["tables"] / "rank_summary.csv",
            lambda frame: frame.group_id.isin(["clean", "primary_joint"]),
        ),
        correlations=_selected_csv(
            paths["tables"] / "rank_correlations.csv",
            lambda frame: frame.group_rhs.eq("primary_joint"),
        ),
    )


def run(config_path):
    cfg = load_config(config_path)
    frozen = require_preregistration(cfg)
    paths = stage_paths(cfg, "full")
    verification = json.loads(
        (paths["logs"] / "independent_verification.json").read_text(encoding="utf-8")
    )
    if (
        verification.get("status") != "passed"
        or verification.get("config_sha256") != cfg["_config_sha256"]
    ):
        raise ValueError(
            "Full independent verification must pass before the final report"
        )
    visual = json.loads(
        (Path(cfg["_results_root"]) / "logs" / "visual_review.json").read_text(
            encoding="utf-8"
        )
    )
    if visual.get("status") != "passed" or set(visual.get("stages", {})) != {
        "pilot",
        "full",
    }:
        raise ValueError(
            "Both stages require an actual visual review before final delivery"
        )
    for stage, reviewed in visual["stages"].items():
        rendered = audit_figures(cfg, stage, stage_paths(cfg, stage))
        expected_pngs = {
            item["path"]: item
            for item in rendered["outputs"]
            if Path(item["path"]).suffix == ".png"
        }
        reviewed_files = reviewed.get("files", [])
        if (
            reviewed.get("status") != "passed"
            or reviewed.get("figure_manifest") != rendered["manifest"]
            or len(reviewed_files) != rendered["figure_pairs"]
            or {item["path"]: item for item in reviewed_files} != expected_pngs
        ):
            raise ValueError(f"{stage} visual approval does not match current figures")
        if (
            stage == "full"
            and rendered["manifest"] != verification["figures"]["manifest"]
        ):
            raise ValueError("Full figures changed after independent verification")
    tables = build_tables(cfg, "full")
    summary = tables["summary"].set_index(
        ["model", "strategy", "group_id", "metric", "outcome"]
    )

    def mean_sd(model, strategy, group, metric, outcome="absolute"):
        row = summary.loc[(model, strategy, group, metric, outcome)]
        text = f"{number(row['mean'])} ± {number(row['std'])}"
        if row.n_undefined:
            text += f"（{int(row.n_seeds)}/{int(row.n_expected)} seeds 有定义）"
        return text

    primary, secondary, nstdb = tables["primary"], tables["secondary"], tables["nstdb"]
    primary_e = primary[primary.lhs.eq("electrode") & primary.rhs.eq("independent_rms")]
    primary_absolute = tables["effects"].query(
        "metric == 'macro_auroc' and outcome == 'absolute'"
    )
    judgments = []
    for question, label in [
        ("unseen_strength", "未见强度"),
        ("unseen_combination", "未见组合"),
    ]:
        chosen = secondary[
            secondary.question.eq(question) & secondary.rhs.eq("independent_rms")
        ]
        judgments.append(
            [
                label,
                f"electrode 相对 independent-RMS 的绝对 AUROC 均值在 {int((chosen.auroc_difference > 0).sum())}/{len(chosen)} 个模型×SNR 单元为正；{int((chosen.positive_seed_count == 5).sum())}/{len(chosen)} 个单元五个训练 seed 全部同向。属于分解后的次要描述性证据，见表中原始差值。",
            ]
        )
    judgments.append(
        [
            "跨模型、跨训练 seed",
            f"联合未见主终点中，electrode−independent-RMS 的 retention 差有 {int(primary_e.positive_seed_count.sum())}/10 个模型×seed 为正；两模型中 {int(((primary_e.point > 0) & (primary_e.p_holm < .05)).sum())}/2 项获得正向 Holm 校正支持，{int((primary_e.patient_ci95_low > 0).sum())}/2 项的条件患者区间完全高于 0。两类区间不是联合区间。",
        ]
    )
    judgments.append(
        [
            "clean 代价是否可接受",
            "不能用本实验自动裁决临床可接受性：没有预设效用/非劣界值。下表同时报告 clean 指标及同 seed clean-only 减策略的代价；正值代表损失，负值代表改善。retention 必须连同绝对 noisy AUROC 和 clean 代价解读。",
        ]
    )
    nstdb_main = nstdb[nstdb.rhs.eq("independent_rms") & ~nstdb.pressure]
    judgments.append(
        [
            "NSTDB 是否同向",
            f"在 electrode 结构、非 0 dB 的 NSTDB 敏感性单元中，electrode−independent-RMS 绝对 AUROC 差为正的有 {int((nstdb_main.auroc_difference > 0).sum())}/{len(nstdb_main)}；完整按模型、噪声类型和强度拆分，不能将源域混杂结果升级为真实采集机制的因果结论。",
        ]
    )
    model_direction = (
        "同向"
        if primary_e.point.gt(0).all() or primary_e.point.lt(0).all()
        else "不完全一致或含零效应"
    )
    model_profile = "；".join(
        f"{r.model}: Δretention={number(r.point, 6)}，正向 seed={r.positive_seed_count}/5"
        for r in primary_e.itertuples(index=False)
    )
    judgments.append(
        [
            "是否依赖模型",
            f"本次两架构的 electrode−independent-RMS 主终点点估计方向{model_direction}；{model_profile}。这些是固定架构的观察，不把“一个显著、一个不显著”当作显著的模型间差异。未预注册模型×策略交互检验，也没有覆盖其他架构、训练预算或数据集。",
        ]
    )
    broad_support = bool(
        (
            (primary_e.point > 0)
            & (primary_e.p_holm < 0.05)
            & (primary_e.patient_ci95_low > 0)
            & primary_e.positive_seed_count.eq(5)
        ).all()
    )
    absolute_support = (
        primary_absolute[
            primary_absolute.lhs.eq("electrode")
            & primary_absolute.rhs.eq("independent_rms")
        ]
        .point.gt(0)
        .all()
    )
    judgment = (
        "在本数据、两种固定网络和该扰动协议内，有一致的机制增强比较证据；仍不等于普适训练优势或临床有效性。"
        if broad_support and absolute_support
        else "不足以把机制一致增强表述为跨模型普适训练优势；主要结论应限于该评估协议与具体条件下的实测比较，不能择优展示个别模型或 seed。"
    )
    judgments.append(["评价协议还是普适训练优势", judgment])

    out = Path(cfg["_phase2_root"]) / "reports"
    out.mkdir(parents=True, exist_ok=True)
    report = [
        "# 第二阶段实测报告：四种训练策略对未见采集扰动的泛化",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}。本报告只使用已完成并独立核验的 full 结果；pilot 单列为技术验证，不用于调参。",
        "## 1. 七项判断",
        markdown_table(["问题", "基于实测数据的回答"], judgments),
        "## 2. 冻结设计与执行完整性",
        f"科学配置 SHA-256：`{cfg['_config_sha256']}`。预注册时间：`{frozen['created_at_utc']}`。导联矩阵 SHA-256：`{cfg['matrix_sha256']}`。第一阶段的代码、缓存、模型及既有预测保持原样。",
        "完整阶段为 4 策略 × 2 模型 × 5 个训练 seed（17、29、43、101、202），每次固定 25 epochs。官方 folds 1–8 / 9 / 10：17,084 / 2,146 / 2,158 条；无五类目标标签的 411 条继续排除。原 ResNet 545,717 参数，原 TCN 134,885 参数。",
        "AdamW：lr=0.001、weight decay=0.0001、batch=128；FP32，无 AMP/TF32、梯度裁剪、scheduler 或 early stopping。每条记录每 epoch 一次曝光。沿用第一阶段去均值 mV 数据，先加噪后除以固定完整训练集 RMS 0.2254905871835822 mV，不逐导联归一化。",
        markdown_table(
            ["策略", "clean", "independent-RMS", "electrode"],
            [
                ["clean_only", "100%", "0%", "0%"],
                ["independent_rms", "50%", "50%", "0%"],
                ["electrode", "50%", "0%", "50%"],
                ["mixed", "50%", "25%", "25%"],
            ],
        ),
        "训练只用 0.5–40 Hz Gaussian，20/10 dB 各半；增强每轮动态产生且 mutually exclusive。初始化、shuffle、增强决策、噪声源流隔离，同一 seed 的初始化和记录顺序配对；增强不依赖模型名称。训练/测试 base 分别为 10001 和 20001–20005，且另有 domain/stage 键。",
        "方案冲突事前裁决：组合采用 §13 的明确不重叠列表；V1/V6 单电极已在训练支持内，不称为未见组合；不启用随机 2–4 电极而泄漏测试组合。遵循第一阶段实际无梯度裁剪设置。主比较采用 §15 的联合未见 retention；没有事后增加显著性家族或虚构 clean 临床可接受界值。",
        "十个训练组合：RA+LA+LL（概率 1/4）；单 V1–V6 及 RA+LA、LA+LL、RA+LL（各 1/12）。十个未见组合：RA、LA、LL、RA+V1、LA+V6、LL+V4、RA+LA+V1+V6、RA+V2+V5、LA+V3+V6、LL+V1+V4。已见支持集合的测试汇总按组合等权，不声称其边际混合概率等于训练分布。",
        "所有 epoch checkpoint 均保存。用 clean-validation Macro-AUROC 选最早最大值；固定轮数训练结束后，最优 checkpoint 的五个 F1 阈值只在 clean validation 确定一次，全部测试条件共用。没有 noisy-validation 选模或测试集阈值重调。",
        "### 2.1 测试矩阵与实际输入",
        "完整测试包含 1,711 个共享 case：Gaussian 的（10 训练组合 + 10 未见组合 + 全电极）× 5 SNR（20/15/10/5/0）× 3 结构 × 5 次噪声，共 1,575；NSTDB 的 bw/ma/em × 20/10/0 dB × 3 结构 × 5 次全电极回放，共 135；另加 clean。40 checkpoint 共 68,440 份逐记录预测。",
        "三种结构为 independent-RMS、electrode、covariance-matched。缓存只能在该阶段全部训练结束后创建。保存 float32 的 0 dB 基噪声及固定组合公式，对每个 SNR 的实际归一化输入计算哈希；评估时精确重建并共用一个 GPU 输入张量。逐记录实际 SNR、逐导联 RMS 和协方差匹配误差随缓存交付。",
        "15 dB 是未训练过的插值强度；5 dB 为更强的外推强度。0 dB 只作压力测试，不进入确认性主终点。75% mixed 和 noisy-validation 两个可选分析不执行。",
        "## 3. 确认性联合未见主终点",
        "主组：electrode 测试结构、十个未见组合、15/5 dB、五个测试 base，共 100 case。每个 checkpoint 先计算并等权平均 case AUROC，再除以自身 clean AUROC + 1e−12；不是概率 ensemble，也不把重复条件当新患者。AUROC retention 未作机会水平校正。",
        "六项比较为两模型 ×（electrode−clean_only、electrode−independent_rms、mixed−electrode）。对五个配对训练 seed 的 retention 差作双侧 t 检验，六项共同 Holm 校正。AP 同时作为主要性能描述指标，但未追加其确认性假设检验。",
    ]
    report.append(
        markdown_table(
            ["模型", "策略", "主组 AUROC", "主组 AP", "主组 F1", "AUROC retention"],
            [
                [
                    model,
                    LABELS[strategy],
                    *[
                        mean_sd(model, strategy, "primary_joint", metric)
                        for metric in ("macro_auroc", "macro_ap", "macro_f1")
                    ],
                    mean_sd(
                        model, strategy, "primary_joint", "macro_auroc", "retention"
                    ),
                ]
                for model in cfg["models"]
                for strategy in cfg["strategies"]
            ],
        )
    )
    report.append(
        "均值 ± 样本 SD；正文数值适度舍入，全部未舍入值保存在对应 CSV。下列对比使用同一个 seed 的两策略相减，而不是独立样本比较。"
    )
    primary_rows = []
    for row in primary.itertuples(index=False):
        absolute = primary_absolute.loc[
            primary_absolute.model.eq(row.model)
            & primary_absolute.lhs.eq(row.lhs)
            & primary_absolute.rhs.eq(row.rhs)
        ].iloc[0]
        primary_rows.append(
            [
                row.model,
                f"{LABELS[row.lhs]} − {LABELS[row.rhs]}",
                number(row.point),
                bounds(row.ci95_low, row.ci95_high),
                bounds(row.patient_ci95_low, row.patient_ci95_high),
                probability(row.p_raw),
                probability(row.p_holm),
                number(row.dz, 3),
                f"{row.positive_seed_count}/5",
                number(absolute.point),
            ]
        )
    report += [
        markdown_table(
            [
                "模型",
                "配对比较",
                "Δretention",
                "训练 seed t95%",
                "条件患者 95%",
                "原始 p",
                "Holm p",
                "配对 dz",
                "正向 seed",
                "绝对 noisy ΔAUROC",
            ],
            primary_rows,
        ),
        "训练 seed 区间与患者区间回答不同问题。患者区间固定训练 checkpoint 和测试噪声；即使该区间不跨 0，也不能替代训练随机性和六项 Holm 结果。配对 t 推断依赖训练 seed 差值独立、近似正态；仅五个 seed 无法可靠检验该近似，区间与功效均受小样本限制。retention 的改善不能单独排除 clean 分母变差所致的表象。",
        "![主终点及次要绝对/drop 对比](../results/figures/full/primary_contrast_intervals.png)",
        "## 4. Clean 性能及代价",
    ]
    clean_rows = []
    cost_rows = []
    for model in cfg["models"]:
        for strategy in cfg["strategies"]:
            clean_rows.append(
                [
                    model,
                    LABELS[strategy],
                    *[
                        mean_sd(model, strategy, "clean", metric)
                        for metric in (
                            "macro_auroc",
                            "macro_ap",
                            "macro_f1",
                            "brier",
                            "ece",
                        )
                    ],
                ]
            )
            cost_rows.append(
                [
                    model,
                    LABELS[strategy],
                    *[
                        mean_sd(model, strategy, "clean", metric, "clean_cost")
                        for metric in ("macro_auroc", "macro_ap", "macro_f1")
                    ],
                ]
            )
    report += [
        markdown_table(
            ["模型", "策略", "AUROC", "AP", "F1", "Brier ↓", "ECE ↓"], clean_rows
        ),
        "所有 ± 为五个训练 seed 的样本 SD，不是置信区间。",
        markdown_table(
            ["模型", "策略", "clean AUROC 代价", "clean AP 代价", "clean F1 代价"],
            cost_rows,
        ),
        "正代价表示比同 seed 的 clean-only 更差。AUROC 的条件患者区间与逐 seed 值见图及完整 patient_ci.csv；没有临床非劣界值，不能自动称代价可接受。",
        "![Clean 性能](../results/figures/full/clean_absolute.png)",
        "![Clean 代价](../results/figures/full/clean_clean_cost.png)",
        "## 5. 分离未见强度与未见组合",
        "未见强度：使用训练组合支持，检查 15/5 dB；未见组合：使用十个 held-out 组合，检查训练已见的 20/10 dB。这样不把两个变化混为同一个问题。下列为次要描述性配对差，不增加事后 p 值。",
    ]
    secondary_rows = [
        [
            r.model,
            "未见强度" if r.question == "unseen_strength" else "未见组合",
            r.snr,
            f"electrode − {LABELS[r.rhs]}",
            number(r.auroc_difference),
            number(r.ap_difference),
            number(r.retention_difference),
            f"{r.positive_seed_count}/5",
        ]
        for r in secondary.itertuples(index=False)
    ]
    report += [
        markdown_table(
            [
                "模型",
                "分解问题",
                "dB",
                "比较",
                "ΔAUROC",
                "ΔAP",
                "Δretention",
                "AUROC 正向 seed",
            ],
            secondary_rows,
        ),
        "![全部 Gaussian AUROC 曲线](../results/figures/full/gaussian_macro_auroc_absolute.png)",
        "![全部 Gaussian retention 曲线](../results/figures/full/gaussian_macro_auroc_retention.png)",
        "![十个未见组合 AUROC](../results/figures/full/heldout_heatmap_absolute.png)",
        "![十个未见组合 retention](../results/figures/full/heldout_heatmap_retention.png)",
        "协方差匹配分支为机制消融参照，independent-RMS 控制逐导联能量和零支持。单胸电极条件本身不产生跨多导联相关结构，不能仅靠这类条件判断协方差机制。所有结构/组合/SNR 的原始指标均保存，不只展示有利条件。",
        markdown_table(
            [
                "模型",
                "测试结构",
                "dB",
                "electrode−independent ΔAUROC",
                "Δretention",
                "正向 seed",
            ],
            [
                [
                    r.model,
                    r.condition,
                    r.snr,
                    number(r.auroc_difference),
                    number(r.retention_difference),
                    f"{r.positive_seed_count}/5",
                ]
                for r in tables["mechanism"].itertuples(index=False)
            ],
        ),
        "## 6. 分类、阈值与校准辅助指标",
    ]
    for group, title in [("clean", "Clean"), ("primary_joint", "联合未见主组")]:
        report.append(f"### {title}")
        for columns, metrics in [
            (
                ["Micro-AUROC", "Micro-AP", "Micro-F1", "Brier ↓", "ECE ↓"],
                ["micro_auroc", "micro_ap", "micro_f1", "brier", "ece"],
            ),
            (
                [
                    "Sensitivity",
                    "Specificity",
                    "PPV",
                    "NPV",
                    "标签一致率",
                    "平均概率变化",
                ],
                [
                    "macro_sensitivity",
                    "macro_specificity",
                    "macro_ppv",
                    "macro_npv",
                    "clean_agreement",
                    "probability_shift",
                ],
            ),
        ]:
            report.append(
                markdown_table(
                    ["模型", "策略", *columns],
                    [
                        [
                            model,
                            LABELS[strategy],
                            *[
                                mean_sd(model, strategy, group, metric)
                                for metric in metrics
                            ],
                        ]
                        for model in cfg["models"]
                        for strategy in cfg["strategies"]
                    ],
                )
            )
    report += [
        "Sensitivity/Specificity/PPV/NPV 是五类 macro 比值；若某类分母为 0，macro 保持未定义，并披露有定义训练 seed 数，而不将它填成 0。标签一致率及平均概率变化都相对于同 checkpoint 的 clean 预测。阈值始终来自 clean validation；ECE 和可靠性图没有对 test 拟合校准器。",
        "![ResNet 可靠性图](../results/figures/full/calibration_resnet.png)",
        "![TCN 可靠性图](../results/figures/full/calibration_tcn.png)",
        "### 6.1 逐类 AUROC / AP / F1",
        markdown_table(
            ["条件", "模型", "策略", "类别", "AUROC", "AP", "F1"],
            [
                [
                    group,
                    model,
                    LABELS[strategy],
                    label,
                    *[
                        mean_sd(model, strategy, group, f"{metric}_{label}")
                        for metric in ("auroc", "ap", "f1")
                    ],
                ]
                for group in ("clean", "primary_joint")
                for model in cfg["models"]
                for strategy in cfg["strategies"]
                for label in cfg["class_order"]
            ],
        ),
        "## 7. 训练、患者、噪声三类不确定性",
    ]
    pred = verification["predictions"]
    report += [
        f"患者簇 bootstrap 使用固定 seed 20260919 和 2000 次共同抽样，test 有 {pred['patients']} 个患者、{pred['records']} 条 ECG。抽中一个患者保留其全部 ECG 和抽样重复；所有策略与条件共用相同 multiplicities。已保存每个固定 checkpoint 与固定 seed 指标均值的 AUROC/逐类 AUROC 区间、drop、retention、clean cost 和配对差。",
        "缺类别 draw 保留为 NaN 并计数，绝不补抽；macro 始终要求全部五类。percentile CI 使用有效 draw，并报告有效/无效个数。固定 seed 均值是各模型指标/比值/差值在共同 draw 内的均值，不是先平均概率。",
        "五个测试 noise base 只刻画固定模型的噪声实现敏感性，不视为额外训练。下表先在每个噪声 base 内对训练 seed 的配对 retention 差求均值，再展示五次回放的描述性范围。",
    ]
    noise_summary = tables["noise"][tables["noise"].level.eq("summary")]
    report.append(
        markdown_table(
            ["模型", "比较", "noise 均值", "noise SD", "范围", "正向 noise"],
            [
                [
                    r.model,
                    f"{LABELS[r.lhs]} − {LABELS[r.rhs]}",
                    number(r.point),
                    number(r.std),
                    bounds(r.min, r.max),
                    f"{int(r.positive_noise_count)}/{int(r.n_noise_seeds)}",
                ]
                for r in noise_summary.itertuples(index=False)
            ],
        )
    )
    report += [
        "![噪声重复稳定性](../results/figures/full/noise_realization_stability.png)",
        "![主配对差的噪声重复](../results/figures/full/noise_primary_contrast_stability.png)",
        "## 8. 四策略排名及描述性一致性",
        markdown_table(
            ["模型", "条件", "指标", "策略", "平均 rank", "rank SD"],
            [
                [
                    r.model,
                    r.group_id,
                    r.metric,
                    LABELS[r.strategy],
                    number(r.mean_rank, 2),
                    number(r.std_rank, 2),
                ]
                for r in tables["ranks"].itertuples(index=False)
                if r.metric in ("macro_auroc", "macro_ap")
            ],
        ),
        "rank=1 为最好；AUROC/AP/F1 越高越好，Brier/ECE 越低越好。表中平均 rank 是逐 seed 排名的均值，不是先平均指标再重新排名。仅四种策略；并列平均排名、Spearman/Kendall 均作描述性用途。常量排名导致未定义相关时不填零。",
        markdown_table(
            ["模型", "指标", "clean/主组 Spearman", "Kendall"],
            [
                [r.model, r.metric, number(r.spearman, 3), number(r.kendall, 3)]
                for r in tables["correlations"].itertuples(index=False)
                if str(r.seed) == "mean_fixed_seeds"
            ],
        ),
        "![Clean 逐 seed 排名](../results/figures/full/ranks_individual_seeds_clean.png)",
        "![主组逐 seed 排名](../results/figures/full/ranks_individual_seeds_primary_joint.png)",
        "![SNR 与排名](../results/figures/full/rank_snr_heldout.png)",
        "![Clean/扰动排名相关](../results/figures/full/rank_correlations_clean_noisy.png)",
        "## 9. NSTDB：探索性、有混杂的敏感性分析",
        "bw/ma/em 是真实记录噪声源，但这里将导联空间记录重新用于电极扰动。源/记录域不匹配、全电极压力支持与 Gaussian 主条件不同，因此不是独立的真实采集因果验证。0 dB 单列压力；不能用其替换不显著的预注册主终点。",
        markdown_table(
            [
                "模型",
                "类型",
                "dB",
                "比较",
                "ΔAUROC",
                "ΔAP",
                "Δretention",
                "AUROC 正向 seed",
            ],
            [
                [
                    r.model,
                    r.kind,
                    str(r.snr) + ("（压力）" if r.pressure else ""),
                    f"electrode − {LABELS[r.rhs]}",
                    number(r.auroc_difference),
                    number(r.ap_difference),
                    number(r.retention_difference),
                    f"{r.positive_seed_count}/5",
                ]
                for r in nstdb.itertuples(index=False)
            ],
        ),
        "![全部 NSTDB 结构与类型](../results/figures/full/exploratory_nstdb_sensitivity.png)",
        "## 10. 技术 pilot、执行异常与独立验收",
        "Pilot：4,000 / 1,000 / 1,000 条，seed 17，8 次训练各 10 epochs；118 个共享 case、944 份预测、2000 次患者簇抽样、18 组 PNG/PDF。技术门控只检查固定预算、曝光、噪声强度、流/组合分离及产物，不选择优胜策略，也不改变 full 设置。",
        "第一次监督任务退出码为 58，没有 Python traceback；当时 full 首次 clean-only/ResNet/seed17 仅记录到第17轮。确认没有残留子进程后，保留原尝试日志并从相同注册初始化重新训练完整25轮；后续采用可跨监督会话存续的进程。没有将部分运行充作已完成模型，也没有调整训练预算。所有尝试以 run_status.json 为准。",
        f"独立验收核对 {pred['prediction_hashes_checked']} 份预测哈希；对 clean 和全部主条件独立用 sklearn 重算 {pred['independent_auc_ap_f1_cases']} 组 AUROC/AP/F1，最大绝对误差分别为 {pred['max_macro_auroc_error']:.3g} / {pred['max_macro_ap_error']:.3g} / {pred['max_macro_f1_error']:.3g}。显式复制被抽中患者全部 ECG 的复算覆盖 {pred['explicit_patient_repeat_cases']} 个 condition×draw；逐项重算六个主对比、t 区间、患者区间及 Holm。",
        f"对全部 {pred['inference_replay_checkpoints']} 个 checkpoint 的 clean 与一个主条件各128条记录重新推理（{pred['inference_replay_device']}），最大概率差 {pred['inference_replay_max_probability_error']:.3g}。所有训练 epoch、数据/输入缓存、表格与图源指纹核对；完整图形文件完成解码、PDF头检查及独立目视检查。详见验收 JSON，不把窄范围 smoke 冒充全矩阵验收。",
        "![完整训练与 clean validation 轨迹](../results/figures/full/training_clean_validation_trajectories.png)",
        "## 11. 推断边界",
        "该研究比较的是固定 PTB-XL 划分、两种网络、固定训练预算和明确噪声机制下的采集扰动泛化。合成 Gaussian、电极理想映射和 NSTDB 重用不包含全部真实设备、阻抗、饱和、非线性或临床分布转移。五个训练 seed 的 t 推断样本仍小；患者区间不含训练随机性；五次噪声不是五次独立训练。没有独立外部临床验证。",
        "结论需要同时对照 clean、绝对 noisy 性能、retention、两模型方向和全部训练 seed。不得将局部最好结果、未校正次要比较或探索性 NSTDB 结果升级为普适训练优越性。",
        "## 12. 复现与原始交付",
        "命令及目录见 [README](../README.md)。核心来源：",
        "- [预注册](../results/logs/preregistration.json)；[运行历史](../results/logs/run_status.json)；[实现修订说明](../results/logs/implementation_amendments.json)。",
        "- [完整原始指标](../results/tables/full/metrics.csv)；[每训练 seed 组指标](../results/tables/full/group_seed_metrics.csv)；[均值/SD/t区间](../results/tables/full/seed_summary.csv)。",
        "- [主比较](../results/tables/full/primary_comparisons.csv)；[配对效应](../results/tables/full/paired_effects.csv)；[患者区间](../results/tables/full/patient_ci.csv)；[无效 draw 计数](../results/tables/full/bootstrap_invalid_summary.csv)。",
        "- [患者抽样及分布索引](../results/tables/full/patient_bootstrap/manifest.json)；[测试输入索引](../results/test_inputs/full/manifest.json)；[图及源表索引](../results/figures/full/figure_manifest.json)。",
        "- [完整独立验收](../results/logs/full/independent_verification.json)；[pilot 独立验收](../results/logs/pilot/independent_verification.json)；[目视检查](../results/logs/visual_review.json)；[第一阶段保护复核](../results/logs/phase1_preservation_verified.json)。",
        "补充图：[未见组合 AP/F1](../results/figures/full/supplement_gaussian_heldout_ap_f1.png)、[Brier/ECE](../results/figures/full/supplement_gaussian_heldout_brier_ece.png)。所有图同时提供同名 PDF。",
    ]
    report_path = out / "phase2_final_report.md"
    report_path.write_text("\n\n".join(report) + "\n", encoding="utf-8")
    exports = []
    for name in ("summary", "secondary", "nstdb", "mechanism"):
        filename = paths["tables"] / f"report_{name}.csv"
        tables[name].to_csv(filename, index=False)
        exports.append(file_info(filename))
    sources = [
        paths["tables"] / f"{name}.csv"
        for name in (
            "seed_summary",
            "group_seed_metrics",
            "paired_effects",
            "primary_comparisons",
            "noise_contrasts",
            "rank_summary",
            "rank_correlations",
        )
    ]
    save_json(
        paths["logs"] / "report_manifest.json",
        dict(
            status="completed",
            config_sha256=cfg["_config_sha256"],
            report=file_info(report_path),
            exports=exports,
            sources=[file_info(path) for path in sources],
            generator=file_info(Path(__file__)),
            verification=file_info(paths["logs"] / "independent_verification.json"),
        ),
    )
    print(f"Saved evidence-driven report: {report_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="phase2/configs/phase2_main.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
