# Artifact provenance

All paths below are relative to the original local workspace
`FSEML-rewrite-2026`. Copies were made on 2026-09-08.

| Destination | Original source |
|---|---|
| `logs/` | `log/` (all available `.log` and other non-PID files) |
| `cifar100/available_main_results/` | `ANML-master_20260805/cifar100_main_results/` |
| `cifar100/efficiency/results/` | `ANML-master_20260808/cifar100_efficiency_results/` |
| `cifar100/efficiency/configs/` | JSON configs from `ANML-master_20260808/cifar100_efficiency_models/` |
| `ablation/component_isolation/` | `ANML-master_20260809/ablation_studies/03_component_isolation/`, excluding `.net` checkpoints |
| `reconstruction_fidelity/` | `ANML-master_20260808/reconstruction_fidelity_analysis_20260804/` and `20260509CSV/latent_probe_summary.csv` |
| `core50/` | JSON records from `ANML-master_20260905/core50_heldout_oml_30k/` and `core50_heldout_transferred_30k/` |
| `clinc150/` | JSON records from `ANML-master_20260906/clinc150_heldout_oml_10k/` and `clinc150_heldout_transferred_10k/` |

The repository source outside `artifacts/` originates from
`ANML-master_20260908/`, excluding caches and operating-system metadata.

`expected_results.json` is maintained separately from the archived experiment
artifacts listed above.
