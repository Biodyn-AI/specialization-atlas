#!/usr/bin/env bash
# Pipeline orchestrator for the Head/Layer Specialization Atlas analysis.
#
# Run from the repository root:
#   bash scripts/run_all.sh
#
# Prerequisites:
#   pip install -r requirements.txt
#
# Note: generate_atlas_head_layer_summary.py is excluded because it requires
# atlas .npy arrays not included in this repository. Its pre-computed output
# (data/evidence/head_layer_atlas_correlations.csv) is provided.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Head/Layer Specialization Atlas Pipeline ==="
echo ""

echo "[1/4] Generating workshop tables..."
python scripts/generate_workshop_tables.py \
    --input-dir results/tables \
    --output-dir results/tables
echo "  Done."
echo ""

echo "[2/4] Running adversarial audit..."
python scripts/run_adversarial_audit.py \
    --tables-dir results/tables \
    --correlations-csv data/evidence/head_layer_atlas_correlations.csv \
    --output-dir results/tables
echo "  Done."
echo ""

echo "[3/4] Running extended head biology analysis..."
python scripts/run_extended_head_biology_analysis.py \
    --evidence data/evidence/head_layer_evidence.tsv \
    --top-heads results/tables/top_heads_summary.tsv \
    --audit-perm-global results/tables/audit_permutation_global_fdr.tsv \
    --tables-out results/tables \
    --figures-out results/figures
echo "  Done."
echo ""

echo "[4/4] Running research-grade extensions..."
python scripts/run_research_grade_extensions.py \
    --evidence data/evidence/head_layer_evidence.tsv \
    --tf-summary results/tables/tf_enrichment_summary.tsv \
    --tables-out results/tables \
    --figures-out results/figures
echo "  Done."
echo ""

echo "=== Pipeline complete. ==="
echo "Tables:  results/tables/"
echo "Figures: results/figures/"
