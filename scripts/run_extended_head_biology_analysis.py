#!/usr/bin/env python3
"""Run extended head/layer analyses for publication-grade interpretation.

This script reuses existing edge-level head evidence tables and derives:
1) head usage profiles (including zero-dominant behavior),
2) high-confidence edge enrichment by head with FDR control,
3) bridge tables linking atlas-significant heads to evidence-heavy heads,
4) GO:BP enrichment for selected heads via g:Profiler API,
5) publication-ready figures.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
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


def head_id(layer: int, head: int) -> str:
    return f"{layer}:{head}"


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
    rows = []
    for row in df.itertuples(index=False):
        parsed = parse_top1_head(row.top_head_layers)
        if parsed is None:
            continue
        layer, head = parsed
        rows.append(
            {
                "source": row.source,
                "target": row.target,
                "edge_score": float(row.edge_score),
                "max_head_layer_score": float(row.max_head_layer_score),
                "layer": layer,
                "head": head,
                "head_id": head_id(layer, head),
            }
        )
    out = pd.DataFrame.from_records(rows)
    out["is_nonzero"] = out["max_head_layer_score"] > 0
    return out


def build_edge_deciles(df: pd.DataFrame) -> pd.DataFrame:
    dec = df.copy()
    dec["decile"] = pd.qcut(dec["edge_score"], 10, labels=False, duplicates="drop")
    summary = (
        dec.groupby("decile", as_index=False)
        .agg(
            n_edges=("edge_score", "size"),
            edge_score_min=("edge_score", "min"),
            edge_score_max=("edge_score", "max"),
            edge_score_mean=("edge_score", "mean"),
            max_head_score_mean=("max_head_layer_score", "mean"),
            max_head_score_median=("max_head_layer_score", "median"),
            max_head_score_zero_frac=("max_head_layer_score", lambda x: (x <= 0).mean()),
        )
        .sort_values("decile")
        .reset_index(drop=True)
    )
    return summary


def build_head_profiles(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (layer, head), group in df.groupby(["layer", "head"], sort=True):
        nonzero = group[group["is_nonzero"]]
        records.append(
            {
                "layer": int(layer),
                "head": int(head),
                "head_id": head_id(int(layer), int(head)),
                "n_edges_all": int(len(group)),
                "n_edges_nonzero": int(len(nonzero)),
                "nonzero_fraction": float(len(nonzero) / len(group)) if len(group) else np.nan,
                "mean_edge_score_all": float(group["edge_score"].mean()),
                "mean_edge_score_nonzero": float(nonzero["edge_score"].mean())
                if len(nonzero)
                else np.nan,
                "mean_max_head_score_all": float(group["max_head_layer_score"].mean()),
                "mean_max_head_score_nonzero": float(nonzero["max_head_layer_score"].mean())
                if len(nonzero)
                else np.nan,
                "median_max_head_score_nonzero": float(nonzero["max_head_layer_score"].median())
                if len(nonzero)
                else np.nan,
                "zero_fraction_all": float((group["max_head_layer_score"] <= 0).mean()),
            }
        )
    profile = pd.DataFrame.from_records(records)
    profile = profile.sort_values("n_edges_all", ascending=False).reset_index(drop=True)
    return profile


def build_high_conf_enrichment(
    df: pd.DataFrame,
    min_edges_nonzero: int = 50,
) -> pd.DataFrame:
    nonzero = df[df["is_nonzero"]].copy()
    if nonzero.empty:
        return pd.DataFrame()

    threshold = float(nonzero["edge_score"].quantile(0.90))
    nonzero["is_high_conf"] = nonzero["edge_score"] >= threshold
    total_high = int(nonzero["is_high_conf"].sum())
    total_low = int((~nonzero["is_high_conf"]).sum())

    rows = []
    for (layer, head), group in nonzero.groupby(["layer", "head"], sort=True):
        if len(group) < min_edges_nonzero:
            continue
        a = int(group["is_high_conf"].sum())
        b = int((~group["is_high_conf"]).sum())
        c = total_high - a
        d = total_low - b
        odds, p_hi = fisher_exact([[a, b], [c, d]], alternative="greater")
        _, p_lo = fisher_exact([[a, b], [c, d]], alternative="less")
        rows.append(
            {
                "layer": int(layer),
                "head": int(head),
                "head_id": head_id(int(layer), int(head)),
                "n_edges_nonzero": int(len(group)),
                "high_conf_edges": a,
                "high_conf_fraction": float(a / len(group)),
                "odds_ratio_high_vs_rest": float(odds),
                "p_value_high_enriched": float(p_hi),
                "p_value_low_enriched": float(p_lo),
                "mean_edge_score_nonzero": float(group["edge_score"].mean()),
                "mean_max_head_score_nonzero": float(group["max_head_layer_score"].mean()),
                "edge_score_q90_threshold": threshold,
            }
        )

    out = pd.DataFrame.from_records(rows)
    if out.empty:
        return out
    out["q_value_high_enriched"] = benjamini_hochberg(out["p_value_high_enriched"].to_numpy())
    out["q_value_low_enriched"] = benjamini_hochberg(out["p_value_low_enriched"].to_numpy())
    out["significant_high_enriched_q_lt_0_10"] = out["q_value_high_enriched"] < 0.10
    out["significant_low_enriched_q_lt_0_10"] = out["q_value_low_enriched"] < 0.10
    out = out.sort_values(["q_value_high_enriched", "odds_ratio_high_vs_rest"], ascending=[True, False])
    return out.reset_index(drop=True)


def load_robust_atlas_heads(
    top_heads_path: Path,
    audit_perm_global_path: Path,
) -> pd.DataFrame:
    top = pd.read_csv(top_heads_path, sep="\t")
    perm = pd.read_csv(audit_perm_global_path, sep="\t")
    perm = perm[perm["global_sig_q_lt_0_10"]].copy()
    perm["head_id"] = perm.apply(lambda r: head_id(int(r["layer"]), int(r["head"])), axis=1)

    top["head_id"] = top.apply(lambda r: head_id(int(r["layer"]), int(r["head"])), axis=1)
    merged = top.merge(
        perm[["tissue", "head_id", "q_value_global_bh", "observed_aupr"]],
        on=["tissue", "head_id"],
        how="inner",
    )
    merged = merged[merged["tissue"] != "immune"].copy()
    merged = merged.sort_values(["tissue", "aupr"], ascending=[True, False]).reset_index(drop=True)
    return merged


GENE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


def sanitize_genes(genes: list[str], max_genes: int = 200) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for gene in genes:
        if not isinstance(gene, str):
            continue
        gene = gene.strip()
        if not gene or gene in seen:
            continue
        if not GENE_RE.match(gene):
            continue
        # Drop extremely long synthetic identifiers.
        if len(gene) > 24:
            continue
        seen.add(gene)
        cleaned.append(gene)
        if len(cleaned) >= max_genes:
            break
    return cleaned


def fetch_gprofiler_go_bp(
    genes: list[str],
    timeout: int = 60,
) -> list[dict[str, object]]:
    if len(genes) < 20:
        return []
    url = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
    payload = {
        "organism": "hsapiens",
        "query": genes,
        "sources": ["GO:BP"],
        "user_threshold": 0.05,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    if resp.status_code != 200:
        return []
    data = resp.json()
    return data.get("result", [])


def infer_theme(term_name: str) -> str:
    name = term_name.lower()
    if any(x in name for x in ["immune", "lymphocyte", "immunoglobulin", "antigen"]):
        return "immune_adaptive"
    if any(x in name for x in ["metabolic", "catabolic", "carboxylic", "oxoacid"]):
        return "metabolic_catabolic"
    if any(x in name for x in ["response to stimulus", "response to"]):
        return "stimulus_response"
    if any(x in name for x in ["translation", "ribosome", "ribonucleoprotein"]):
        return "translation_rna"
    return "other"


def build_go_enrichment(
    df: pd.DataFrame,
    high_conf_enrichment: pd.DataFrame,
    robust_atlas_heads: pd.DataFrame,
    max_heads: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nonzero = df[df["is_nonzero"]].copy()
    threshold = float(nonzero["edge_score"].quantile(0.90))
    nonzero = nonzero[nonzero["edge_score"] >= threshold].copy()

    selected_heads: list[str] = []
    selected_heads.extend(
        high_conf_enrichment.loc[
            high_conf_enrichment["significant_high_enriched_q_lt_0_10"],
            "head_id",
        ]
        .head(max_heads)
        .tolist()
    )
    # Ensure canonical atlas heads are included if present.
    selected_heads.extend(robust_atlas_heads["head_id"].drop_duplicates().head(6).tolist())
    selected_heads = list(dict.fromkeys(selected_heads))

    enrich_rows: list[dict[str, object]] = []
    theme_rows: list[dict[str, object]] = []

    for hid in selected_heads:
        head_rows = nonzero[nonzero["head_id"] == hid]
        if head_rows.empty:
            continue

        source_ranked = (
            head_rows.groupby("source")["edge_score"].mean().sort_values(ascending=False).index.tolist()
        )
        source_genes = sanitize_genes(source_ranked, max_genes=200)
        results = fetch_gprofiler_go_bp(source_genes)

        if not results:
            theme_rows.append(
                {
                    "head_id": hid,
                    "n_high_conf_edges": int(len(head_rows)),
                    "n_source_genes_submitted": int(len(source_genes)),
                    "top_term": "",
                    "top_term_p_value": np.nan,
                    "top_term_theme": "no_significant_terms",
                }
            )
            continue

        top_terms = results[:8]
        for rank, term in enumerate(top_terms, start=1):
            enrich_rows.append(
                {
                    "head_id": hid,
                    "rank": rank,
                    "term_id": term.get("native", ""),
                    "term_name": term.get("name", ""),
                    "p_value": float(term.get("p_value", np.nan)),
                    "term_size": int(term.get("term_size", 0)),
                    "intersection_size": int(term.get("intersection_size", 0)),
                    "source": term.get("source", ""),
                }
            )

        top = top_terms[0]
        theme_rows.append(
            {
                "head_id": hid,
                "n_high_conf_edges": int(len(head_rows)),
                "n_source_genes_submitted": int(len(source_genes)),
                "top_term": top.get("name", ""),
                "top_term_p_value": float(top.get("p_value", np.nan)),
                "top_term_theme": infer_theme(str(top.get("name", ""))),
            }
        )

    enrich_df = pd.DataFrame.from_records(enrich_rows)
    theme_df = pd.DataFrame.from_records(theme_rows)
    if not enrich_df.empty:
        enrich_df = enrich_df.sort_values(["head_id", "rank"]).reset_index(drop=True)
    if not theme_df.empty:
        theme_df = theme_df.sort_values("head_id").reset_index(drop=True)
    return enrich_df, theme_df


def plot_head_usage(profile: pd.DataFrame, out_path: Path, top_n: int = 15) -> None:
    plot_df = profile.head(top_n).copy()
    plot_df["n_edges_zero"] = plot_df["n_edges_all"] - plot_df["n_edges_nonzero"]
    plot_df = plot_df.iloc[::-1]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["head_id"], plot_df["n_edges_nonzero"], color="#1f77b4", label="nonzero max score")
    ax.barh(
        plot_df["head_id"],
        plot_df["n_edges_zero"],
        left=plot_df["n_edges_nonzero"],
        color="#d62728",
        alpha=0.75,
        label="zero max score",
    )
    ax.set_xlabel("Number of edges with head as top-1 attribution")
    ax.set_ylabel("Head (layer:head)")
    ax.set_title("Top-1 head usage in edge evidence (all edges)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_high_conf_enrichment(enrich: pd.DataFrame, out_path: Path, top_n: int = 25) -> None:
    if enrich.empty:
        return
    plot_df = enrich.copy()
    plot_df["minus_log10_q"] = -np.log10(np.maximum(plot_df["q_value_high_enriched"], 1e-300))
    plot_df = plot_df.sort_values("minus_log10_q", ascending=False).head(top_n).copy()

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = np.where(plot_df["significant_high_enriched_q_lt_0_10"], "#2ca02c", "#7f7f7f")
    ax.scatter(
        plot_df["odds_ratio_high_vs_rest"],
        plot_df["minus_log10_q"],
        c=colors,
        s=55,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.3,
    )
    for row in plot_df.itertuples(index=False):
        if row.significant_high_enriched_q_lt_0_10:
            ax.text(row.odds_ratio_high_vs_rest + 0.02, row.minus_log10_q, row.head_id, fontsize=8)
    ax.axhline(-math.log10(0.10), color="black", linestyle="--", linewidth=1, label="q=0.10")
    ax.set_xlabel("Odds ratio (high-confidence edge enrichment)")
    ax.set_ylabel(r"$-\log_{10}(q)$")
    ax.set_title("Head enrichment for high-confidence edges (nonzero subset)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extended head biology analysis")
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("data/evidence/head_layer_evidence.tsv"),
        help="Full edge-level head evidence TSV.",
    )
    parser.add_argument(
        "--top-heads",
        type=Path,
        default=Path("results/tables/top_heads_summary.tsv"),
        help="Atlas top-head summary TSV.",
    )
    parser.add_argument(
        "--audit-perm-global",
        type=Path,
        default=Path("results/tables/audit_permutation_global_fdr.tsv"),
        help="Global permutation audit TSV.",
    )
    parser.add_argument(
        "--tables-out",
        type=Path,
        default=Path("results/tables"),
        help="Output directory for TSV tables.",
    )
    parser.add_argument(
        "--figures-out",
        type=Path,
        default=Path("results/figures"),
        help="Output directory for figures.",
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
        raise RuntimeError("No parsable head evidence rows found.")

    deciles = build_edge_deciles(evidence_df)
    profiles = build_head_profiles(evidence_df)
    enrich = build_high_conf_enrichment(evidence_df, min_edges_nonzero=50)

    robust_atlas_heads = load_robust_atlas_heads(
        args.top_heads.resolve(),
        args.audit_perm_global.resolve(),
    )
    bridge = robust_atlas_heads.merge(
        profiles[
            [
                "head_id",
                "n_edges_all",
                "n_edges_nonzero",
                "nonzero_fraction",
                "mean_edge_score_nonzero",
                "mean_max_head_score_nonzero",
                "zero_fraction_all",
            ]
        ],
        on="head_id",
        how="left",
    )
    if not enrich.empty:
        bridge = bridge.merge(
            enrich[
                [
                    "head_id",
                    "high_conf_edges",
                    "high_conf_fraction",
                    "odds_ratio_high_vs_rest",
                    "q_value_high_enriched",
                    "significant_high_enriched_q_lt_0_10",
                ]
            ],
            on="head_id",
            how="left",
        )
    bridge = bridge.sort_values(["tissue", "aupr"], ascending=[True, False]).reset_index(drop=True)

    go_terms, go_themes = build_go_enrichment(
        df=evidence_df,
        high_conf_enrichment=enrich,
        robust_atlas_heads=robust_atlas_heads,
        max_heads=6,
    )

    # Persist tables
    deciles.to_csv(tables_out / "extended_edge_score_deciles.tsv", sep="\t", index=False)
    profiles.to_csv(tables_out / "extended_head_profiles.tsv", sep="\t", index=False)
    enrich.to_csv(tables_out / "extended_head_high_conf_enrichment.tsv", sep="\t", index=False)
    bridge.to_csv(tables_out / "extended_atlas_evidence_bridge.tsv", sep="\t", index=False)
    go_terms.to_csv(tables_out / "extended_go_bp_enrichment_by_head.tsv", sep="\t", index=False)
    go_themes.to_csv(tables_out / "extended_head_theme_summary.tsv", sep="\t", index=False)

    summary = pd.DataFrame(
        [
            {
                "n_edges_total": int(len(evidence_df)),
                "global_zero_fraction": float((evidence_df["max_head_layer_score"] <= 0).mean()),
                "dominant_head_all": str(profiles.iloc[0]["head_id"]),
                "dominant_head_all_n": int(profiles.iloc[0]["n_edges_all"]),
                "dominant_head_all_zero_fraction": float(profiles.iloc[0]["zero_fraction_all"]),
                "n_heads_high_conf_q_lt_0_10": int(
                    enrich["significant_high_enriched_q_lt_0_10"].sum()
                )
                if not enrich.empty
                else 0,
                "n_heads_low_conf_q_lt_0_10": int(
                    enrich["significant_low_enriched_q_lt_0_10"].sum()
                )
                if not enrich.empty
                else 0,
                "n_robust_atlas_heads_in_bridge": int(bridge["head_id"].nunique()),
                "n_head_themes_with_go_signal": int(
                    (go_themes["top_term_theme"] != "no_significant_terms").sum()
                )
                if not go_themes.empty
                else 0,
            }
        ]
    )
    summary.to_csv(tables_out / "extended_summary.tsv", sep="\t", index=False)

    # Persist figures
    plot_head_usage(profiles, figures_out / "extended_top1_head_usage.png", top_n=15)
    plot_high_conf_enrichment(enrich, figures_out / "extended_high_conf_head_enrichment.png", top_n=25)

    print(f"Extended analysis tables written to {tables_out}")
    print(f"Extended analysis figures written to {figures_out}")


if __name__ == "__main__":
    main()
