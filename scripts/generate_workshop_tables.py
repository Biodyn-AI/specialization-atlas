#!/usr/bin/env python3
"""Build workshop-paper summary tables from existing atlas artifacts.

This script is intentionally lightweight: it does not rerun model inference.
It derives publication-focused tables from the already generated atlas outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    """Return BH-adjusted q-values in the original order of p-values."""
    p = np.asarray(list(p_values), dtype=float)
    n = p.size
    if n == 0:
        return np.array([], dtype=float)

    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.empty(n, dtype=float)

    # Walk backward so monotonicity is guaranteed after correction.
    running_min = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        corrected = ranked[idx] * n / rank
        running_min = min(running_min, corrected)
        adjusted_ranked[idx] = min(running_min, 1.0)

    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_ranked
    return adjusted


def load_table(table_dir: Path, name: str) -> pd.DataFrame:
    table_path = table_dir / name
    if not table_path.exists():
        raise FileNotFoundError(f"Missing required table: {table_path}")
    return pd.read_csv(table_path, sep="\t")


def build_top_head_effects(
    baseline: pd.DataFrame,
    top_heads: pd.DataFrame,
) -> pd.DataFrame:
    aggregate = baseline.loc[
        baseline["baseline"] == "aggregate",
        [
            "tissue",
            "aupr_mean",
            "candidate_pairs",
            "candidate_positives",
            "candidate_positive_rate",
            "evaluated_pairs",
        ],
    ].copy()
    aggregate = aggregate.rename(columns={"aupr_mean": "aggregate_aupr"})

    top_idx = top_heads.groupby("tissue")["aupr"].idxmax()
    winners = top_heads.loc[
        top_idx,
        [
            "tissue",
            "layer",
            "head",
            "aupr",
            "aupr_bootstrap_ci_low",
            "aupr_bootstrap_ci_high",
            "bootstrap_top_k_freq",
        ],
    ].copy()
    winners = winners.rename(columns={"aupr": "top_head_aupr"})

    merged = winners.merge(aggregate, on="tissue", how="inner")
    merged["delta_aupr"] = merged["top_head_aupr"] - merged["aggregate_aupr"]
    merged["relative_gain_pct"] = np.where(
        merged["aggregate_aupr"] > 0,
        100.0 * merged["delta_aupr"] / merged["aggregate_aupr"],
        np.nan,
    )
    merged["top_head_bootstrap_ci_width"] = (
        merged["aupr_bootstrap_ci_high"] - merged["aupr_bootstrap_ci_low"]
    )
    return merged.sort_values("tissue").reset_index(drop=True)


def build_permutation_tables(permutation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored_chunks: list[pd.DataFrame] = []

    for tissue, tissue_df in permutation.groupby("tissue", sort=False):
        local = tissue_df.copy()
        local["q_value_bh"] = benjamini_hochberg(local["p_value"].to_numpy())
        local["significant_p_lt_0_05"] = local["p_value"] < 0.05
        local["significant_q_lt_0_10"] = local["q_value_bh"] < 0.10
        local["aupr_minus_perm_mean"] = local["observed_aupr"] - local["perm_mean"]
        local["z_like"] = np.where(
            local["perm_std"] > 0,
            local["aupr_minus_perm_mean"] / local["perm_std"],
            np.nan,
        )
        scored_chunks.append(local)

    scored = pd.concat(scored_chunks, ignore_index=True)
    scored = scored.sort_values(["tissue", "q_value_bh", "p_value"]).reset_index(drop=True)

    summary = (
        scored.groupby("tissue", as_index=False)
        .agg(
            n_heads_tested=("head", "size"),
            n_heads_p_lt_0_05=("significant_p_lt_0_05", "sum"),
            n_heads_q_lt_0_10=("significant_q_lt_0_10", "sum"),
            min_p_value=("p_value", "min"),
            min_q_value=("q_value_bh", "min"),
            median_observed_aupr=("observed_aupr", "median"),
            median_perm_aupr=("perm_mean", "median"),
            median_z_like=("z_like", "median"),
        )
        .sort_values("tissue")
        .reset_index(drop=True)
    )

    return scored, summary


def build_conservation_summary(table_dir: Path) -> pd.DataFrame:
    overlap_specs = [
        ("head_overlap_pairwise.tsv", 10),
        ("head_overlap_top25_pairwise.tsv", 25),
        ("head_overlap_top50_pairwise.tsv", 50),
    ]
    rows: list[dict[str, float | int]] = []

    for file_name, top_k in overlap_specs:
        overlap_df = load_table(table_dir, file_name).copy()
        overlap_df["enrichment_vs_expected"] = (
            overlap_df["intersection"] / overlap_df["expected_overlap"]
        )
        rows.append(
            {
                "top_k": top_k,
                "n_tissue_pairs": int(len(overlap_df)),
                "mean_jaccard": float(overlap_df["jaccard"].mean()),
                "median_jaccard": float(overlap_df["jaccard"].median()),
                "max_jaccard": float(overlap_df["jaccard"].max()),
                "min_p_value": float(overlap_df["p_value"].min()),
                "n_pairs_p_lt_0_05": int((overlap_df["p_value"] < 0.05).sum()),
                "mean_enrichment_vs_expected": float(
                    overlap_df["enrichment_vs_expected"].mean()
                ),
                "max_enrichment_vs_expected": float(
                    overlap_df["enrichment_vs_expected"].max()
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("top_k").reset_index(drop=True)


def build_readiness_tables(
    top_head_effects: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    sweep_summary: pd.DataFrame,
    ablation_effects: pd.DataFrame,
    min_candidate_pairs: int,
    min_sig_heads: int,
    min_spearman: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    readiness = top_head_effects.merge(
        permutation_summary[
            [
                "tissue",
                "n_heads_q_lt_0_10",
                "min_p_value",
                "min_q_value",
            ]
        ],
        on="tissue",
        how="left",
    )
    readiness = readiness.merge(
        sweep_summary[["tissue", "mean_spearman", "mean_jaccard", "n_pairs"]],
        on="tissue",
        how="left",
    )
    readiness = readiness.merge(
        ablation_effects[["tissue", "mean_diff", "cohens_d"]].rename(
            columns={"mean_diff": "ablation_mean_diff_aupr"}
        ),
        on="tissue",
        how="left",
    )

    readiness["gate_non_degenerate_candidates"] = (
        readiness["candidate_pairs"] >= min_candidate_pairs
    )
    readiness["gate_positive_top_head_delta"] = readiness["delta_aupr"] > 0
    readiness["gate_permutation_support"] = (
        readiness["n_heads_q_lt_0_10"].fillna(0) >= min_sig_heads
    )
    readiness["gate_rank_stability"] = (
        readiness["mean_spearman"].fillna(-np.inf) >= min_spearman
    )
    readiness["gate_ablation_direction"] = readiness["ablation_mean_diff_aupr"] > 0

    gate_cols = [
        "gate_non_degenerate_candidates",
        "gate_positive_top_head_delta",
        "gate_permutation_support",
        "gate_rank_stability",
        "gate_ablation_direction",
    ]
    readiness["readiness_score"] = readiness[gate_cols].sum(axis=1).astype(int)
    readiness["ready_for_primary_claims"] = readiness["readiness_score"] >= 4
    readiness = readiness.sort_values("tissue").reset_index(drop=True)

    overall = pd.DataFrame(
        [
            {
                "n_tissues_total": int(len(readiness)),
                "n_ready_for_primary_claims": int(
                    readiness["ready_for_primary_claims"].sum()
                ),
                "n_non_degenerate_candidates": int(
                    readiness["gate_non_degenerate_candidates"].sum()
                ),
                "n_positive_top_head_delta": int(
                    readiness["gate_positive_top_head_delta"].sum()
                ),
                "n_with_permutation_support": int(
                    readiness["gate_permutation_support"].sum()
                ),
                "n_with_rank_stability": int(readiness["gate_rank_stability"].sum()),
                "n_with_positive_ablation_direction": int(
                    readiness["gate_ablation_direction"].sum()
                ),
            }
        ]
    )

    return readiness, overall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate publication-oriented summary tables from "
            "results/tables."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/tables"),
        help="Directory containing source TSV tables from atlas analysis.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tables"),
        help="Directory where derived paper tables will be written.",
    )
    parser.add_argument(
        "--min-candidate-pairs",
        type=int,
        default=200,
        help="Minimum candidate pairs required to mark a tissue as non-degenerate.",
    )
    parser.add_argument(
        "--min-sig-heads",
        type=int,
        default=3,
        help="Minimum q<0.10 heads required for permutation support.",
    )
    parser.add_argument(
        "--min-spearman",
        type=float,
        default=0.60,
        help="Minimum mean Spearman sweep stability required for readiness.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_table(input_dir, "baseline_summary.tsv")
    top_heads = load_table(input_dir, "top_heads_summary.tsv")
    permutation = load_table(input_dir, "permutation_summary.tsv")
    sweep_summary = load_table(input_dir, "sweep_summary.tsv")
    ablation_effects = load_table(input_dir, "ablation_effects.tsv")

    top_head_effects = build_top_head_effects(baseline, top_heads)
    permutation_scored, permutation_summary = build_permutation_tables(permutation)
    conservation_summary = build_conservation_summary(input_dir)
    readiness, readiness_overall = build_readiness_tables(
        top_head_effects=top_head_effects,
        permutation_summary=permutation_summary,
        sweep_summary=sweep_summary,
        ablation_effects=ablation_effects,
        min_candidate_pairs=args.min_candidate_pairs,
        min_sig_heads=args.min_sig_heads,
        min_spearman=args.min_spearman,
    )

    top_head_effects.to_csv(output_dir / "paper_top_head_effects.tsv", sep="\t", index=False)
    permutation_scored.to_csv(output_dir / "paper_permutation_fdr.tsv", sep="\t", index=False)
    permutation_summary.to_csv(
        output_dir / "paper_permutation_fdr_summary.tsv", sep="\t", index=False
    )
    conservation_summary.to_csv(
        output_dir / "paper_conservation_multik_summary.tsv", sep="\t", index=False
    )
    readiness.to_csv(output_dir / "paper_readiness_by_tissue.tsv", sep="\t", index=False)
    readiness_overall.to_csv(
        output_dir / "paper_readiness_overall.tsv", sep="\t", index=False
    )

    print(f"Wrote derived tables to {output_dir}")


if __name__ == "__main__":
    main()
