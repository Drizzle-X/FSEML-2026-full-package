# CORe50 server runbook

## Data

Use the official cropped 128×128 RGB archive listed on the
[CORe50 dataset page](https://vlomonaco.github.io/core50/index.html#download).
The upstream download currently resolves to
`http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip`.
The official project describes CORe50 as CC BY 4.0, but this reproduction
repository does not redistribute the multi-gigabyte image archive.

Download and extract the archive so that the data root is either the extracted
`core50_128x128/` directory itself or its parent. The experiment also uses the
official `LUP.pkl`, `labels.pkl` and `paths.pkl` metadata copied into
`vendor/core50_official/` from the
[official source repository](https://github.com/vlomonaco/core50).

The large metadata files are intentionally excluded from Git. Recreate them on the server with:

```bash
bash setup_core50_metadata.sh
```

The official page specifies 164,866 cropped RGB images. It does not publish a
cryptographic checksum for the ZIP archive, so this repository does not claim
an unverifiable archive hash. For provenance, calculate and record the SHA-256
of the downloaded archive:

```bash
shasum -a 256 /path/to/core50_128x128.zip
```

Set the extracted image directory explicitly:

```bash
export CORE50_DATA_ROOT=/absolute/path/to/core50_128x128
```

The directory must make paths such as `s4/o16/C_04_16_077.png` resolvable beneath it.

Expected structure:

```text
core50_128x128/
├── s1/
│   ├── o1/C_01_01_000.png
│   ├── ...
│   └── o50/
├── ...
└── s11/
    ├── o1/
    └── o50/C_11_50_*.png
```

After extracting, run the reproducibility verifier. It checks SHA-256 values
for the official metadata, the canonical 164,866-entry path-list digest, the
existence of every expected image, and one decoded 128×128 RGB image from each
session/object directory:

```bash
python verify_core50_data.py --data-root "$CORE50_DATA_ROOT"
```

For the strictest check, decode and validate all images:

```bash
python verify_core50_data.py --data-root "$CORE50_DATA_ROOT" --image-check all
```

Published verification values:

```text
paths.pkl   SHA-256 073e9a900fa558a31e9b9264b0cad7a3339c7b6d14ced0f602200ada1d2b694f
LUP.pkl     SHA-256 6aca5cd9af1ee9850bd0e6f25e865be2719a888faa2f708dd1142af16e5fa545
labels.pkl  SHA-256 123dcd893c3e4187d51712c1e7c1c809e4ddb06d56eb3aa4aaf42a7ad4f5c46c
ordered official paths joined with LF
            SHA-256 f4c03f616b52d073f3253d719b10d65a928d009c1feb3cc90603d9189a81d4b8
```

## Paper protocol

Manuscript Table III uses the official **NICv2-79** temporal and batch
ordering. The 50 object classes are divided into 30 mutually disjoint
meta-training classes and 20 unseen meta-test classes. The 20 test classes are
evaluated as four consecutive 5-way tasks. This is the canonical CORe50
protocol for the reproduction package.

## Local validation completed

- Official CORe50 metadata and protocol manifests load successfully.
- A synthetic two-step SRN/FSSAE + CPN meta-update completed.
- Decoder replay triggered, sampled stored latents and completed backward propagation.
- Python compilation passed for the loader, protocol builder and experiment entry point.

## Full paper run

```bash
# Run from the root of this cloned repository.
bash setup_core50_metadata.sh
export CORE50_DATA_ROOT=/absolute/path/to/core50_128x128
export PYTHON_BIN="$(command -v python)"
bash run_core50_server.sh
```

The run used for Table III must select NICv2-79 and the 30/20 disjoint class
split described above.

## Optional NC diagnostic

For a less severe task-incremental comparison, use the official CORe50 New
Classes (`nc`) scenario. It contains nine training tasks (10 classes initially,
then 5 new classes per task). Shuffle examples within each task and restrict
prediction to the evaluated task's classes, matching the task-aware convention
used by the legacy CIFAR100/Omniglot evaluation:

```bash
python run_core50_experiment.py \
  --data-root "$CORE50_DATA_ROOT" --scenario nc --run 0 --seed 9 \
  --initial-epochs 10 --learning-rate 1e-4 --shuffle-experiences \
  --prediction-class-scope task --no-replay --eval-every 1 \
  --output-dir core50_nc_task_aware_results
```

This protocol should be reported as task-aware NC and must not be described as
task-free NICv2. A matched `--prediction-class-scope seen` result can be added
as a stricter class-incremental supplement.

This diagnostic is not the protocol used for the manuscript Table III result.

Batch 0 is a shared ten-class initialization phase: it is shuffled for five
epochs by default and fills the replay buffer, but replay sampling is disabled.
From batch 1 onward, the official NIC order is preserved, every experience is
single-pass, and replay sampling is enabled for FSEML-ER.

CORe50 uses 50 class-indexed replay partitions by default. Using the inherited
ten-partition modulo rule is invalid for the official initial classes
`0, 5, 10, ..., 45`: it aliases all ten classes into only two partitions and
silently reduces a 1000-item buffer to 200 usable entries. Training losses are
also restricted to the global set of classes observed so far, while evaluation
remains task-free over that same seen-class set.

## Optional NICv2-391 stress test

NICv2-391 is a longer-stream extension, not the manuscript protocol. FSE
replay computes per-sample gradient-norm importance, so this extension is
substantially more expensive than CIFAR100. First run:

```bash
python run_core50_experiment.py \
  --data-root "$CORE50_DATA_ROOT" --scenario nicv2_391 --run 0 --seed 9 \
  --max-batches 3 --eval-every 1 --initial-epochs 5 \
  --replay --output-dir core50_smoke_results
```

Only after this smoke test succeeds should an optional NICv2-391 extension be
launched. Its results must not be labeled as the Table III CORe50 result.

## Expected outputs

```text
core50_results/<run>.pt
core50_results/<run>_metrics.json
```

Before paper use, add/run ER, OML and preferably DER++ under the same official streams and byte budget. The current server script produces the FSEML/FSEML-ER core comparison.
