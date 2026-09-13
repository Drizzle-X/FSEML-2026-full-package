# Reproduction package status

This repository is assembled from `ANML-master_20260908` as the maintained
code base. Historical artifacts are included under `artifacts/` for audit and
result reconstruction. Expected metrics are listed in `expected_results.json`.

## Included now

- Unified Omniglot, CIFAR-100, CORE50, and CLINC150 source code.
- Pinned Python requirements and a Conda environment specification.
- Dataset protocols and CORE50/CLINC150 preparation documentation.
- Manuscript-aligned protocol descriptions: CORe50 NICv2-79 with a disjoint
  30/20 class split and four 5-way test tasks; CLINC150 with a disjoint 6/4
  domain split (90/60 intents) and twelve reported 5-way test tasks.
- Available text logs from the local experiment workspace.
- Available CIFAR-100 main-result, efficiency, reconstruction, and component
  isolation JSON/CSV artifacts.
- Available CORE50 and CLINC150 structured evaluation records.
- A machine-readable provenance manifest with SHA-256 hashes.
- A table/command/result index, manuscript expectations, a standard-library
  result validator, and dependency-free smoke tests.
- Five deterministic main-experiment seeds, all 15 CIFAR-100 split manifests,
  a shared Table I/II configuration, a 90-command runner, and a five-run
  Omniglot/CIFAR-100 mean/standard-deviation aggregator. Omniglot evaluation
  uses training-side learning-rate selection and emits its full accuracy
  matrix with ACC, FM, and LA.
- Five deterministic CORe50 and CLINC150 protocol manifests, a five-seed Table
  III runner, and an ACC/FM/LA mean-and-standard-deviation aggregator.
- An MIT source license and CFF citation stub, with publication metadata marked
  for completion before archival release.

## Deliberate exclusions

- Dataset files: licenses prohibit or discourage redistribution and they are
  large. Follow the dataset preparation guides instead.
- Model checkpoints (`.net`, `.pt`, `.pth`, `.ckpt`): these make the repository
  hundreds of megabytes larger and are not required to inspect the training
  implementation. Release checkpoints separately if desired.
- Python caches, notebook checkpoints, editor files, PID files, and duplicate
  copies of identical historical directories.

## Expected results

`expected_results.json` lists FSEML and FSEML-ER metrics for Tables I–III.
`PAPER_RESULTS_MAPPING.md` maps those table numbers to the relevant commands
and result identifiers.
