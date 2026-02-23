# Head/Layer Specialization Atlas for scGPT Across Tissues

This repository contains data, analysis scripts, and manuscript materials for the paper *"Head/Layer Specialization Atlas for scGPT Across Tissues: Extended Empirical Analysis and Biological Interpretation"*.

We present a head/layer specialization atlas for scGPT across kidney, lung, immune, and external Krasnow lung data. The analysis integrates tissue-level recovery metrics, global false-discovery controls, edge-level head attribution analysis (240,150 edges), threshold-robust enrichment tests, and biological program interpretation via GO enrichment. The central finding is a heterogeneous head ecosystem: tissue-optimized heads, biologically reusable immune/metabolic program heads, and low-information fallback heads.

## Repository Structure

```
head-layer-specialization-atlas/
├── scripts/               Analysis pipeline scripts (Python)
├── configs/               Experiment configuration files (documentation)
├── data/evidence/         Pre-computed edge-level evidence tables
├── results/
│   ├── tables/            All analysis output tables (TSV)
│   └── figures/           All analysis output figures (PNG)
└── manuscript/            Anonymous submission manuscript (LaTeX)
```

## Quick Start

```bash
pip install -r requirements.txt
bash scripts/run_all.sh
```

This re-derives all analysis tables and figures from the included evidence data. Output tables and figures in `results/` are overwritten in-place with identical results, serving as a verification step.

## Prerequisites

- Python >= 3.9
- Packages: see `requirements.txt` (numpy, pandas, scipy, matplotlib, seaborn, requests)
- Internet connection required for GO enrichment (g:Profiler API) in `run_extended_head_biology_analysis.py`

## Analysis Pipeline

The pipeline runs four scripts in sequence. Each script uses `argparse` and can be run individually with custom paths.

| Step | Script | Description | Key Inputs | Key Outputs |
|------|--------|-------------|------------|-------------|
| 1 | `generate_workshop_tables.py` | Derives publication-focused summary tables (top-head effects, permutation FDR, conservation, readiness) | `results/tables/baseline_summary.tsv`, `top_heads_summary.tsv`, `permutation_summary.tsv`, `sweep_summary.tsv`, `ablation_effects.tsv`, `head_overlap_*.tsv` | `paper_top_head_effects.tsv`, `paper_permutation_fdr.tsv`, `paper_readiness_*.tsv` |
| 2 | `run_adversarial_audit.py` | Stress-tests publication claims under global FDR, family-wise overlap correction, and correlation significance | Step 1 outputs + `data/evidence/head_layer_atlas_correlations.csv` | `audit_*.tsv` (9 files) |
| 3 | `run_extended_head_biology_analysis.py` | Edge-level head profiles, high-confidence enrichment (Fisher exact + FDR), GO:BP enrichment, bridge analysis | `data/evidence/head_layer_evidence.tsv` + Step 2 audit tables | `extended_*.tsv` (7 files) + figures |
| 4 | `run_research_grade_extensions.py` | Threshold-robust enrichment sweep (top 20/15/10/5%) and TF-level recall-delta analysis | `data/evidence/head_layer_evidence.tsv` + `tf_enrichment_summary.tsv` | `extended_high_conf_*.tsv`, `extended_tf_*.tsv` + figures |

### Atlas correlation script (not in pipeline)

`generate_atlas_head_layer_summary.py` computes atlas-vs-evidence correlations from raw attention arrays (`.npy`). These arrays (264 MB each, 3 checkpoints) are not included in this repository. The pre-computed output `data/evidence/head_layer_atlas_correlations.csv` is provided instead. To re-run this script, provide the atlas `.npy` arrays via the `--atlas-root` argument.

## Configuration Files

The `configs/` directory contains YAML configurations that document the original experiment setup (attention extraction, candidate-space evaluation, causal interventions, perturbation validation). These are provided for provenance and reproducibility documentation. Absolute paths in these files have been replaced with `<ORIGINAL_WORKSPACE>/` placeholders.

## Data Provenance

- **Tabula Sapiens**: Kidney, lung, and immune subsets from the Tabula Sapiens consortium.
- **Krasnow lung**: External Smart-seq2 lung dataset for independent validation.
- **Perturbation data**: Dixit (2016) and Adamson (2016) CRISPR perturbation experiments for causal validation.
- **Regulatory references**: TRRUST and DoRothEA transcription factor databases.
- **GO enrichment**: g:Profiler API (requires internet connection at runtime).

## Manuscript

The anonymized manuscript is in `manuscript/main.tex` with all referenced figures in `manuscript/figures/`. It compiles with standard `pdflatex`.

## License

MIT
