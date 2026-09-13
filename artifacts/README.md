# Artifact layout

- `logs/`: locally available textual experiment logs, preserved for audit.
- `cifar100/available_main_results/`: structured outputs from the latest
  locally available CIFAR-100 result collection.
- `cifar100/efficiency/`: no-replay, raw-replay, and decoder-replay metrics.
- `ablation/component_isolation/`: Group A configuration and result records.
- `reconstruction_fidelity/`: quantitative reconstruction/replay analysis.
- `core50/` and `clinc150/`: available extension-study JSON records.
- `MANIFEST.sha256`: content hashes and original source paths.

Expected metrics are stored in the repository-level `expected_results.json`;
this directory preserves supporting experiment files.
