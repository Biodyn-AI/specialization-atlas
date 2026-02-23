#!/usr/bin/env python3
"""Adversarial statistical audit for the head/layer atlas paper.

The goal is to stress-test publication claims under stricter controls:
- Global FDR (across tissues) for permutation tests.
- Family-wise overlap correction across all top-k pairwise tests.
- Correlation significance and confidence intervals for atlas-vs-evidence checks.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    """Return BH-adjusted q-values in original p-value order."""
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


def load_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, sep="\t")


def fisher_z_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Fisher z confidence interval for Pearson correlation."""
    if n <= 3 or not np.isfinite(r):
        return (np.nan, np.nan)

    r = max(min(r, 0.999999), -0.999999)
    z = 0.5 * math.log((1.0 + r) / (1.0 - r))
    se = 1.0 / math.sqrt(n - 3)
    z_crit = 1.959963984540054  # two-sided alpha=0.05
    lo = z - z_crit * se
    hi = z + z_crit * se

    def inv_fisher(value: float) -> float:
        e2 = math.exp(2.0 * value)
        return (e2 - 1.0) / (e2 + 1.0)

    return inv_fisher(lo), inv_fisher(hi)


def pearson_p_value_from_r(r: float, n: int) -> float:
    """Two-sided p-value for Pearson r using normal approximation on Fisher z."""
    if n <= 3 or not np.isfinite(r):
        return np.nan
    r = max(min(r, 0.999999), -0.999999)
    z = 0.5 * math.log((1.0 + r) / (1.0 - r))
    se = 1.0 / math.sqrt(n - 3)
    z_stat = z / se
    # Normal CDF via erf: Phi(x) = 0.5 * (1 + erf(x / sqrt(2))).
    tail = 1.0 - 0.5 * (1.0 + math.erf(abs(z_stat) / math.sqrt(2.0)))
    return min(max(2.0 * tail, 0.0), 1.0)


def odds_ratio(a: int, b: int, c: int, d: int, correction: float = 0.5) -> float:
    """Haldane-corrected odds ratio."""
    return ((a + correction) * (d + correction)) / ((b + correction) * (c + correction))


def run_audit(
    tables_dir: Path,
    correlations_csv: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Core tables
    permutation = load_tsv(tables_dir / "permutation_summary.tsv")
    overlap_k10 = load_tsv(tables_dir / "head_overlap_pairwise.tsv")
    overlap_k25 = load_tsv(tables_dir / "head_overlap_top25_pairwise.tsv")
    overlap_k50 = load_tsv(tables_dir / "head_overlap_top50_pairwise.tsv")
    readiness = load_tsv(tables_dir / "paper_readiness_by_tissue.tsv")
    top_effects = load_tsv(tables_dir / "paper_top_head_effects.tsv")

    # 1) Global permutation FDR (all tissues together)
    perm_global = permutation.copy()
    perm_global["q_value_global_bh"] = benjamini_hochberg(perm_global["p_value"])
    perm_global["global_sig_q_lt_0_10"] = perm_global["q_value_global_bh"] < 0.10
    perm_global["global_sig_q_lt_0_05"] = perm_global["q_value_global_bh"] < 0.05
    perm_global["aupr_minus_perm_mean"] = (
        perm_global["observed_aupr"] - perm_global["perm_mean"]
    )
    perm_global["z_like"] = np.where(
        perm_global["perm_std"] > 0,
        perm_global["aupr_minus_perm_mean"] / perm_global["perm_std"],
        np.nan,
    )

    perm_global_summary = (
        perm_global.groupby("tissue", as_index=False)
        .agg(
            n_heads=("head", "size"),
            n_sig_global_q_lt_0_10=("global_sig_q_lt_0_10", "sum"),
            n_sig_global_q_lt_0_05=("global_sig_q_lt_0_05", "sum"),
            min_global_q=("q_value_global_bh", "min"),
            median_z_like=("z_like", "median"),
        )
        .sort_values("tissue")
        .reset_index(drop=True)
    )

    # 2) Overlap correction across all k-level pairwise tests (6*3 = 18 tests)
    overlap_all = pd.concat(
        [
            overlap_k10.assign(top_k=10),
            overlap_k25.assign(top_k=25),
            overlap_k50.assign(top_k=50),
        ],
        ignore_index=True,
    )
    overlap_all["q_value_global_bh"] = benjamini_hochberg(overlap_all["p_value"])
    overlap_all["sig_raw_p_lt_0_05"] = overlap_all["p_value"] < 0.05
    overlap_all["sig_global_q_lt_0_10"] = overlap_all["q_value_global_bh"] < 0.10
    overlap_all["sig_global_q_lt_0_05"] = overlap_all["q_value_global_bh"] < 0.05
    overlap_all["enrichment_vs_expected"] = (
        overlap_all["intersection"] / overlap_all["expected_overlap"]
    )

    overlap_global_summary = (
        overlap_all.groupby("top_k", as_index=False)
        .agg(
            n_pairs=("tissue_a", "size"),
            n_sig_raw_p_lt_0_05=("sig_raw_p_lt_0_05", "sum"),
            n_sig_global_q_lt_0_10=("sig_global_q_lt_0_10", "sum"),
            n_sig_global_q_lt_0_05=("sig_global_q_lt_0_05", "sum"),
            min_raw_p=("p_value", "min"),
            min_global_q=("q_value_global_bh", "min"),
            mean_jaccard=("jaccard", "mean"),
            mean_enrichment=("enrichment_vs_expected", "mean"),
        )
        .sort_values("top_k")
        .reset_index(drop=True)
    )

    # 3) Correlation significance audit for atlas-vs-evidence alignment
    if not correlations_csv.exists():
        raise FileNotFoundError(f"Missing correlations CSV: {correlations_csv}")
    corr = pd.read_csv(correlations_csv)
    corr = corr.copy()

    # Use evidence_nonzero_heads as sample size proxy for each correlation estimate.
    corr["n_heads"] = corr["evidence_nonzero_heads"].astype(int)
    corr["pearson_p_approx"] = [
        pearson_p_value_from_r(r, n) for r, n in zip(corr["pearson"], corr["n_heads"])
    ]
    corr["pearson_ci_low"] = np.nan
    corr["pearson_ci_high"] = np.nan
    for idx, row in corr.iterrows():
        lo, hi = fisher_z_ci(float(row["pearson"]), int(row["n_heads"]))
        corr.at[idx, "pearson_ci_low"] = lo
        corr.at[idx, "pearson_ci_high"] = hi

    corr["pearson_q_global_bh"] = benjamini_hochberg(corr["pearson_p_approx"])
    corr["pearson_sig_q_lt_0_10"] = corr["pearson_q_global_bh"] < 0.10
    corr["pearson_sig_q_lt_0_05"] = corr["pearson_q_global_bh"] < 0.05

    corr_summary = (
        corr.groupby(["evidence_set", "mean_type"], as_index=False)
        .agg(
            n_rows=("dataset", "size"),
            mean_pearson=("pearson", "mean"),
            min_pearson=("pearson", "min"),
            max_pearson=("pearson", "max"),
            mean_spearman=("spearman", "mean"),
            n_sig_q_lt_0_10=("pearson_sig_q_lt_0_10", "sum"),
            n_sig_q_lt_0_05=("pearson_sig_q_lt_0_05", "sum"),
        )
        .sort_values(["evidence_set", "mean_type"])
        .reset_index(drop=True)
    )

    # 4) Degeneracy and readiness risk table
    risk = readiness[
        [
            "tissue",
            "candidate_pairs",
            "candidate_positive_rate",
            "delta_aupr",
            "n_heads_q_lt_0_10",
            "mean_spearman",
            "ablation_mean_diff_aupr",
            "readiness_score",
            "ready_for_primary_claims",
        ]
    ].copy()
    risk["degenerate_candidate_space"] = risk["candidate_pairs"] < 50
    risk["high_base_rate"] = risk["candidate_positive_rate"] > 0.25
    risk["zero_or_negative_delta"] = risk["delta_aupr"] <= 0
    risk["no_permutation_support"] = risk["n_heads_q_lt_0_10"] == 0
    risk["low_rank_stability"] = risk["mean_spearman"].fillna(-np.inf) < 0.60
    risk["nonpositive_ablation_direction"] = (
        risk["ablation_mean_diff_aupr"].fillna(-np.inf) <= 0
    )
    risk_flags = [
        "degenerate_candidate_space",
        "high_base_rate",
        "zero_or_negative_delta",
        "no_permutation_support",
        "low_rank_stability",
        "nonpositive_ablation_direction",
    ]
    risk["n_risk_flags"] = risk[risk_flags].sum(axis=1).astype(int)
    risk = risk.sort_values("tissue").reset_index(drop=True)

    # 5) Claim robustness summary
    robust_claims = []

    # Claim: head specialization positive (non-degenerate tissues).
    non_deg = top_effects[top_effects["candidate_pairs"] >= 200]
    n_non_deg = len(non_deg)
    n_positive = int((non_deg["delta_aupr"] > 0).sum())
    robust_claims.append(
        {
            "claim_id": "C1_top_head_gain_non_degenerate",
            "description": "Top-head AUPR exceeds aggregate in non-degenerate tissues.",
            "numerator": n_positive,
            "denominator": n_non_deg,
            "estimate": n_positive / n_non_deg if n_non_deg else np.nan,
            "passes_adversarial_gate": n_non_deg > 0 and n_positive == n_non_deg,
            "notes": "Uses candidate_pairs>=200 filter.",
        }
    )

    # Claim: permutation support survives global FDR.
    perm_support_by_tissue = perm_global.groupby("tissue")["global_sig_q_lt_0_10"].sum()
    n_tissues_with_perm = int((perm_support_by_tissue > 0).sum())
    robust_claims.append(
        {
            "claim_id": "C2_permutation_global_fdr",
            "description": "At least one head remains significant per tissue under global BH q<0.10.",
            "numerator": n_tissues_with_perm,
            "denominator": int(perm_support_by_tissue.size),
            "estimate": n_tissues_with_perm / perm_support_by_tissue.size,
            "passes_adversarial_gate": n_tissues_with_perm >= 3,
            "notes": "Includes immune tissue.",
        }
    )

    # Claim: cross-tissue conservation remains significant after full correction.
    n_sig_overlap_global = int((overlap_all["q_value_global_bh"] < 0.10).sum())
    robust_claims.append(
        {
            "claim_id": "C3_conservation_global_fdr",
            "description": "Cross-tissue overlap significance survives BH q<0.10 across all k/pairs.",
            "numerator": n_sig_overlap_global,
            "denominator": int(len(overlap_all)),
            "estimate": n_sig_overlap_global / len(overlap_all),
            "passes_adversarial_gate": n_sig_overlap_global > 0,
            "notes": "18 tests: 6 tissue pairs x 3 top-k levels.",
        }
    )

    # Claim: evidence alignment positive for mean_nonzero correlations.
    nonzero_corr = corr[corr["mean_type"] == "mean_nonzero"]
    n_nonzero_pos = int((nonzero_corr["pearson"] > 0).sum())
    robust_claims.append(
        {
            "claim_id": "C4_evidence_alignment_nonzero",
            "description": "Pearson correlation is positive for mean_nonzero atlas-evidence summaries.",
            "numerator": n_nonzero_pos,
            "denominator": int(len(nonzero_corr)),
            "estimate": n_nonzero_pos / len(nonzero_corr),
            "passes_adversarial_gate": n_nonzero_pos == len(nonzero_corr),
            "notes": "Uses head_layer_atlas_correlations.csv.",
        }
    )

    claims_df = pd.DataFrame(robust_claims)

    # 6) Save all audit outputs
    perm_global.to_csv(output_dir / "audit_permutation_global_fdr.tsv", sep="\t", index=False)
    perm_global_summary.to_csv(
        output_dir / "audit_permutation_global_fdr_summary.tsv", sep="\t", index=False
    )
    overlap_all.to_csv(output_dir / "audit_overlap_global_fdr.tsv", sep="\t", index=False)
    overlap_global_summary.to_csv(
        output_dir / "audit_overlap_global_fdr_summary.tsv", sep="\t", index=False
    )
    corr.to_csv(output_dir / "audit_evidence_correlation_significance.tsv", sep="\t", index=False)
    corr_summary.to_csv(
        output_dir / "audit_evidence_correlation_summary.tsv", sep="\t", index=False
    )
    risk.to_csv(output_dir / "audit_tissue_risk_flags.tsv", sep="\t", index=False)
    claims_df.to_csv(output_dir / "audit_claim_robustness.tsv", sep="\t", index=False)

    # Single-file executive summary for manuscript ingestion.
    summary = pd.DataFrame(
        [
            {
                "n_perm_tests": int(len(perm_global)),
                "n_perm_sig_global_q_lt_0_10": int(perm_global["global_sig_q_lt_0_10"].sum()),
                "n_overlap_tests": int(len(overlap_all)),
                "n_overlap_sig_global_q_lt_0_10": int(
                    (overlap_all["q_value_global_bh"] < 0.10).sum()
                ),
                "n_evidence_corr_tests": int(len(corr)),
                "n_evidence_corr_sig_global_q_lt_0_10": int(
                    (corr["pearson_q_global_bh"] < 0.10).sum()
                ),
                "n_tissues_ready_primary_claims": int(
                    risk["ready_for_primary_claims"].sum()
                ),
                "n_tissues_with_3plus_risk_flags": int((risk["n_risk_flags"] >= 3).sum()),
                "n_claims_passing_adversarial_gate": int(
                    claims_df["passes_adversarial_gate"].sum()
                ),
                "n_claims_tested": int(len(claims_df)),
            }
        ]
    )
    summary.to_csv(output_dir / "audit_summary.tsv", sep="\t", index=False)

    print(f"Adversarial audit outputs written to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adversarial statistical audit.")
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("results/tables"),
        help="Directory containing atlas summary TSV tables.",
    )
    parser.add_argument(
        "--correlations-csv",
        type=Path,
        default=Path("data/evidence/head_layer_atlas_correlations.csv"),
        help="CSV with atlas-vs-evidence correlation summaries.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tables"),
        help="Directory where audit tables will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_audit(
        tables_dir=args.tables_dir.resolve(),
        correlations_csv=args.correlations_csv.resolve(),
        output_dir=args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
