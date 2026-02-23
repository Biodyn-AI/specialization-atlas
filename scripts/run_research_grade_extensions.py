#!/usr/bin/env python3
"""Research-grade extension analyses for the head/layer specialization paper.

This script adds two robustness/interpretability layers:
1) high-confidence threshold sweep for head enrichment (edge-level evidence),
2) TF-level recall-delta summaries for biological interpretation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    n = p.size
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.empty(n, dtype=float)
    running_min = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        corrected = ranked[idx] * n / rank
        running_min = min(running_min, corrected)
        adjusted_ranked[idx] = min(running_min, 1.0)
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_ranked
    return adjusted


def parse_top1_head(raw: str) -> tuple[int, int] | None:
    try:
        entries = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not entries:
        return None
    first = entries[0]
    return int(first["layer"]), int(first["head"])


def load_edge_head_table(evidence_path: Path) -> pd.DataFrame:
    df = pd.read_csv(evidence_path, sep="\t")
    rows: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        parsed = parse_top1_head(row.top_head_layers)
        if parsed is None:
            continue
        layer, head = parsed
        rows.append(
            {
                "edge_score": float(row.edge_score),
                "max_head_layer_score": float(row.max_head_layer_score),
                "layer": int(layer),
                "head": int(head),
                "head_id": f"{layer}:{head}",
            }
        )
    out = pd.DataFrame.from_records(rows)
    out["is_nonzero"] = out["max_head_layer_score"] > 0
    return out


def build_threshold_sweep(
    evidence_df: pd.DataFrame,
    quantiles: list[float],
    min_edges_nonzero: int,
) -> pd.DataFrame:
    nonzero = evidence_df[evidence_df["is_nonzero"]].copy()
    if nonzero.empty:
        return pd.DataFrame()

    all_rows: list[dict[str, object]] = []
    for q in quantiles:
        threshold = float(nonzero["edge_score"].quantile(q))
        run = nonzero.copy()
        run["is_high_conf"] = run["edge_score"] >= threshold
        total_high = int(run["is_high_conf"].sum())
        total_low = int((~run["is_high_conf"]).sum())

        rows: list[dict[str, object]] = []
        for (layer, head), group in run.groupby(["layer", "head"], sort=True):
            if len(group) < min_edges_nonzero:
                continue
            a = int(group["is_high_conf"].sum())
            b = int((~group["is_high_conf"]).sum())
            c = total_high - a
            d = total_low - b
            odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
            rows.append(
                {
                    "quantile": float(q),
                    "quantile_label": f"top{int(round((1 - q) * 100)):02d}%",
                    "edge_score_threshold": threshold,
                    "layer": int(layer),
                    "head": int(head),
                    "head_id": f"{layer}:{head}",
                    "n_edges_nonzero": int(len(group)),
                    "high_conf_edges": int(a),
                    "high_conf_fraction": float(a / len(group)),
                    "odds_ratio_high_vs_rest": float(odds_ratio),
                    "p_value_high_enriched": float(p_value),
                }
            )
        if not rows:
            continue
        one = pd.DataFrame.from_records(rows)
        one["q_value_high_enriched"] = benjamini_hochberg(one["p_value_high_enriched"].to_numpy())
        one["significant_q_lt_0_10"] = one["q_value_high_enriched"] < 0.10
        one = one.sort_values(["q_value_high_enriched", "odds_ratio_high_vs_rest"], ascending=[True, False])
        all_rows.extend(one.to_dict("records"))

    out = pd.DataFrame.from_records(all_rows)
    if out.empty:
        return out
    return out.reset_index(drop=True)


def build_threshold_consensus(sweep_df: pd.DataFrame) -> pd.DataFrame:
    if sweep_df.empty:
        return pd.DataFrame()

    by_head = (
        sweep_df.groupby(["layer", "head", "head_id"], as_index=False)
        .agg(
            n_thresholds_tested=("quantile", "nunique"),
            n_thresholds_significant=("significant_q_lt_0_10", "sum"),
            min_q_value=("q_value_high_enriched", "min"),
            max_odds_ratio=("odds_ratio_high_vs_rest", "max"),
            mean_odds_ratio=("odds_ratio_high_vs_rest", "mean"),
            mean_high_conf_fraction=("high_conf_fraction", "mean"),
            max_high_conf_fraction=("high_conf_fraction", "max"),
        )
        .sort_values(
            ["n_thresholds_significant", "min_q_value", "mean_odds_ratio"],
            ascending=[False, True, False],
        )
        .reset_index(drop=True)
    )
    return by_head


def build_tf_tables(tf_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tf = pd.read_csv(tf_path, sep="\t")
    tf["head_id"] = tf["layer"].astype(int).astype(str) + ":" + tf["head"].astype(int).astype(str)

    tf_ranked = tf.sort_values(
        ["tissue", "recall_delta_vs_aggregate", "recall"],
        ascending=[True, False, False],
    ).copy()
    tf_ranked["rank_within_tissue"] = tf_ranked.groupby("tissue").cumcount() + 1

    tf_top = tf_ranked[tf_ranked["rank_within_tissue"] <= 10].copy()
    tf_top = tf_top.reset_index(drop=True)

    tissue_stats = (
        tf.groupby("tissue", as_index=False)
        .agg(
            n_rows=("tf", "size"),
            n_positive_delta=("recall_delta_vs_aggregate", lambda s: int((s > 0).sum())),
            max_delta=("recall_delta_vs_aggregate", "max"),
            mean_delta=("recall_delta_vs_aggregate", "mean"),
            n_unique_heads=("head_id", "nunique"),
            n_unique_tfs=("tf", "nunique"),
        )
        .sort_values("max_delta", ascending=False)
        .reset_index(drop=True)
    )
    return tf_top, tissue_stats


def plot_threshold_heatmap(sweep_df: pd.DataFrame, out_path: Path, max_heads: int = 20) -> None:
    if sweep_df.empty:
        return
    consensus = build_threshold_consensus(sweep_df)
    keep_heads = consensus.head(max_heads)["head_id"].tolist()
    plot_df = sweep_df[sweep_df["head_id"].isin(keep_heads)].copy()
    if plot_df.empty:
        return

    plot_df["minus_log10_q"] = -np.log10(np.clip(plot_df["q_value_high_enriched"], 1e-300, 1.0))
    pivot = plot_df.pivot_table(
        index="head_id",
        columns="quantile_label",
        values="minus_log10_q",
        aggfunc="max",
        fill_value=0.0,
    )
    # Stable display order by significance strength.
    ordered_heads = (
        consensus.set_index("head_id")
        .loc[pivot.index]
        .sort_values(["n_thresholds_significant", "min_q_value"], ascending=[False, True])
        .index
    )
    ordered_cols = sorted(pivot.columns, key=lambda x: int(x.replace("top", "").replace("%", "")), reverse=True)
    pivot = pivot.loc[ordered_heads, ordered_cols]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.35 * len(pivot))))
    sns.heatmap(pivot, cmap="mako", linewidths=0.3, cbar_kws={"label": "-log10(q)"}, ax=ax)
    ax.set_title("High-confidence enrichment robustness across score thresholds")
    ax.set_xlabel("High-confidence definition")
    ax.set_ylabel("Head (layer:head)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_threshold_consensus(consensus_df: pd.DataFrame, out_path: Path, top_n: int = 12) -> None:
    if consensus_df.empty:
        return
    plot_df = consensus_df.head(top_n).iloc[::-1].copy()
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9.5, 6))
    colors = np.where(plot_df["n_thresholds_significant"] >= 3, "#2ca02c", "#1f77b4")
    ax.barh(plot_df["head_id"], plot_df["n_thresholds_significant"], color=colors, alpha=0.9)
    ax.set_xlabel("Number of threshold settings with q < 0.10")
    ax.set_ylabel("Head (layer:head)")
    ax.set_title("Consensus high-confidence heads across thresholds")
    ax.set_xlim(0, plot_df["n_thresholds_tested"].max() + 0.8)
    for row in plot_df.itertuples(index=False):
        ax.text(row.n_thresholds_significant + 0.05, row.head_id, f"OR~{row.mean_odds_ratio:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_tf_delta_distribution(tf_path: Path, out_path: Path) -> None:
    tf = pd.read_csv(tf_path, sep="\t")
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sns.boxplot(
        data=tf,
        x="tissue",
        y="recall_delta_vs_aggregate",
        ax=ax,
        color="#9ecae1",
        fliersize=2,
    )
    sns.stripplot(
        data=tf,
        x="tissue",
        y="recall_delta_vs_aggregate",
        ax=ax,
        color="#08519c",
        size=3,
        alpha=0.7,
        jitter=0.15,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("TF-level recall delta versus aggregate baseline")
    ax.set_xlabel("Tissue")
    ax.set_ylabel("Recall delta (top-head minus aggregate)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_tf_top_deltas(tf_top_df: pd.DataFrame, out_path: Path, top_n: int = 15) -> None:
    if tf_top_df.empty:
        return
    plot_df = tf_top_df.sort_values("recall_delta_vs_aggregate", ascending=False).head(top_n).copy()
    plot_df["label"] = (
        plot_df["tissue"].astype(str)
        + " | "
        + plot_df["head_id"].astype(str)
        + " | "
        + plot_df["tf"].astype(str)
    )
    plot_df = plot_df.iloc[::-1]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    palette = {"lung": "#1f77b4", "kidney": "#2ca02c", "external_krasnow_lung": "#ff7f0e"}
    colors = [palette.get(t, "#7f7f7f") for t in plot_df["tissue"]]
    ax.barh(plot_df["label"], plot_df["recall_delta_vs_aggregate"], color=colors, alpha=0.9)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Recall delta versus aggregate baseline")
    ax.set_ylabel("Tissue | Head | TF")
    ax.set_title("Top TF-specific recall gains by specialized heads")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research-grade extension analyses")
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("data/evidence/head_layer_evidence.tsv"),
        help="Edge-level head evidence TSV.",
    )
    parser.add_argument(
        "--tf-summary",
        type=Path,
        default=Path("results/tables/tf_enrichment_summary.tsv"),
        help="TF enrichment summary TSV.",
    )
    parser.add_argument(
        "--tables-out",
        type=Path,
        default=Path("results/tables"),
        help="Output directory for generated tables.",
    )
    parser.add_argument(
        "--figures-out",
        type=Path,
        default=Path("results/figures"),
        help="Output directory for generated figures.",
    )
    parser.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=[0.80, 0.85, 0.90, 0.95],
        help="Quantiles that define high-confidence thresholds.",
    )
    parser.add_argument(
        "--min-edges-nonzero",
        type=int,
        default=50,
        help="Minimum number of nonzero edges per head for sweep tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables_out = args.tables_out.resolve()
    figures_out = args.figures_out.resolve()
    tables_out.mkdir(parents=True, exist_ok=True)
    figures_out.mkdir(parents=True, exist_ok=True)

    evidence_df = load_edge_head_table(args.evidence.resolve())
    if evidence_df.empty:
        raise RuntimeError("No parsable rows in head evidence table.")

    sweep = build_threshold_sweep(
        evidence_df=evidence_df,
        quantiles=args.quantiles,
        min_edges_nonzero=args.min_edges_nonzero,
    )
    consensus = build_threshold_consensus(sweep)
    sweep.to_csv(tables_out / "extended_high_conf_threshold_sweep.tsv", sep="\t", index=False)
    consensus.to_csv(tables_out / "extended_high_conf_consensus.tsv", sep="\t", index=False)

    tf_top, tf_tissue_stats = build_tf_tables(args.tf_summary.resolve())
    tf_top.to_csv(tables_out / "extended_tf_delta_top.tsv", sep="\t", index=False)
    tf_tissue_stats.to_csv(tables_out / "extended_tf_tissue_stats.tsv", sep="\t", index=False)

    plot_threshold_heatmap(sweep, figures_out / "extended_high_conf_threshold_heatmap.png", max_heads=20)
    plot_threshold_consensus(consensus, figures_out / "extended_high_conf_consensus.png", top_n=12)
    plot_tf_delta_distribution(args.tf_summary.resolve(), figures_out / "extended_tf_delta_distribution.png")
    plot_tf_top_deltas(tf_top, figures_out / "extended_tf_delta_top.png", top_n=15)

    print(f"Research-grade extension tables written to {tables_out}")
    print(f"Research-grade extension figures written to {figures_out}")


if __name__ == "__main__":
    main()
