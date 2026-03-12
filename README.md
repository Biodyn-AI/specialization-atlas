# Head/Layer Specialization Atlas for scGPT Across Tissues

This repository contains data, analysis scripts, and manuscript materials for the paper:

> **Head/Layer Specialization Atlas for scGPT Across Tissues: Extended Empirical Analysis and Biological Interpretation**
> Ihor Kendiukhov

We present a comprehensive head/layer specialization atlas for scGPT spanning kidney, lung, immune, and external Krasnow lung datasets from the Tabula Sapiens consortium. The analysis integrates tissue-level recovery metrics, global false-discovery controls, edge-level head attribution analysis (240,150 edges), threshold-robust enrichment tests, and biological program interpretation via GO enrichment.

## Key Findings

| Finding | Details |
|---------|---------|
| **Head ecosystem** | Three functional types: fallback (8:0), immune/adaptive (0:3, 0:5, 0:1), metabolic (9:1) |
| **Edge-level analysis** | 240,150 edges; head 8:0 dominates volume (54%) but carries 0.02% nonzero signal |
| **Threshold robustness** | Heads 0:3 and 0:5 significant at all 4 tested thresholds; 9:1 at 3/4 |
| **Biological programs** | Immune heads: adaptive immunity, B-cell receptor signaling; Metabolic head: small-molecule catabolism |
| **TF-level gains** | Lung tissue: 19/25 TFs with positive head-specific delta; PPARG +0.417 at heads 4:3/4:4 |
| **Checkpoint replication** | Head 0:3 replicates in whole-human checkpoint (4/4 thresholds, q = 1.15×10⁻¹⁴) |

## Repository Structure

```
specialization-atlas/
├── scripts/               Analysis pipeline scripts (Python)
│   ├── run_all.sh         Run complete pipeline
│   ├── generate_workshop_tables.py
│   ├── run_adversarial_audit.py
│   ├── run_extended_head_biology_analysis.py
│   ├── run_research_grade_extensions.py
│   └── generate_atlas_head_layer_summary.py
├── configs/               Experiment configuration files (YAML)
├── data/
│   └── evidence/          Pre-computed edge-level evidence tables
├── results/
│   ├── tables/            All analysis output tables (TSV)
│   └── figures/           All analysis output figures (PNG)
├── manuscript/
│   ├── main.tex           Manuscript (LaTeX)
│   ├── main.pdf           Compiled manuscript
│   └── figures/           Manuscript figures
├── requirements.txt       Python dependencies
└── LICENSE                MIT License
```

## Quick Start

```bash
git clone https://github.com/Biodyn-AI/specialization-atlas.git
cd specialization-atlas
pip install -r requirements.txt
bash scripts/run_all.sh
```

This re-derives all analysis tables and figures from the included evidence data. Output tables and figures in `results/` are overwritten in-place with identical results, serving as a reproducibility verification step.

## Prerequisites

- Python >= 3.9
- Packages: see `requirements.txt` (numpy, pandas, scipy, matplotlib, seaborn, requests)
- Internet connection required for GO enrichment (g:Profiler API) in `run_extended_head_biology_analysis.py`

## Analysis Pipeline

The pipeline runs four scripts in sequence. Each script uses `argparse` and can be run individually with custom paths.

| Step | Script | Description |
|------|--------|-------------|
| 1 | `generate_workshop_tables.py` | Publication-focused summary tables (top-head effects, permutation FDR, readiness) |
| 2 | `run_adversarial_audit.py` | Stress-tests claims under global FDR and family-wise correction |
| 3 | `run_extended_head_biology_analysis.py` | Edge-level profiles, Fisher enrichment, GO:BP analysis, bridge analysis |
| 4 | `run_research_grade_extensions.py` | Threshold-robust sweep (top 20/15/10/5%) and TF-level recall-delta analysis |

### Atlas correlation script (not in pipeline)

`generate_atlas_head_layer_summary.py` computes atlas-vs-evidence correlations from raw attention arrays (`.npy`). These arrays (~264 MB each, 3 checkpoints) are not included in this repository. The pre-computed output `data/evidence/head_layer_atlas_correlations.csv` is provided instead.

## Data Provenance

- **Tabula Sapiens**: Kidney, lung, and immune subsets ([Jones et al., Science 2022](https://doi.org/10.1126/science.abl4896))
- **Krasnow lung**: External Smart-seq2 lung dataset for independent validation
- **Perturbation data**: [Dixit et al., Cell 2016](https://doi.org/10.1016/j.cell.2016.11.038) and [Adamson et al., Cell 2016](https://doi.org/10.1016/j.cell.2016.11.048) CRISPR screens
- **Regulatory references**: TRRUST v2 and DoRothEA transcription factor databases
- **GO enrichment**: g:Profiler API (requires internet)

## Citation

If you use this atlas or code in your work, please cite:

```bibtex
@article{kendiukhov2026atlas,
  title={Head/Layer Specialization Atlas for scGPT Across Tissues:
         Extended Empirical Analysis and Biological Interpretation},
  author={Kendiukhov, Ihor},
  year={2026}
}
```

## License

MIT
