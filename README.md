# FSEML reproducibility package

This repository is the public reproduction package for **FSEML: A Continuous
Task Self-adaptive Optimization Method with High Autonomy**. It contains the
training and evaluation code, fixed configurations, protocol manifests,
expected manuscript results, validation tools, and available artifacts.

The required reproduction scope is the FSEML and FSEML-ER rows in manuscript
Tables I–III:

| Table | Dataset and setting | Methods | Runs |
|---|---|---|---:|
| I | Omniglot-1 | FSEML, FSEML-ER | 5 |
| II | CIFAR100-5/10/20 | FSEML, FSEML-ER | 5 |
| III | CORe50 NICv2-79 | FSEML, FSEML-ER | 5 |
| III | CLINC150 | FSEML, FSEML-ER | 5 |

`expected_results.json` provides machine-readable expected metrics.
`PAPER_RESULTS_MAPPING.md` gives the detailed table–command–result mapping.

## Repository contents

```text
configs/                         fixed main-experiment configuration
cifar100_protocols/              five-seed CIFAR100 task manifests
core50_heldout_splits/           CORe50 held-out split manifest
clinc150_protocols/              CLINC150 domain protocol manifests
datasets/                        dataset and continual-stream loaders
model/                           FSEML, FSSAE, CPN, and replay code
artifacts/                       logs, metrics, analyses, and provenance
tests/                           minimum package and protocol tests
mrcl_classification.py           Omniglot/CIFAR100 meta-training
```

## Environment

Python 3.10 is recommended. The pinned environment was validated with Python
3.10.18, PyTorch 2.7.1, and torchvision 0.22.1.

```bash
conda env create -f environment.yml
conda activate fseml-reproduction
```

Alternatively:

```bash
python3 -m pip install -r requirements.txt
```

For CUDA systems, install the PyTorch build matching the host CUDA runtime
before the remaining dependencies. A CUDA-capable GPU is strongly recommended
for complete training runs.

## Shared experiment policy

Tables I–III use five independent seeds: `9, 19, 29, 39, 49`. For a given seed,
FSEML and FSEML-ER use the same class split and task order. The shared settings
are stored in `configs/paper_main_five_runs.json`, including 20,000
meta-updates, meta learning rate 0.0005, inner-loop learning rate 0.1, and ten
inner steps. FSEML-ER uses decoder replay with a 1,000-item buffer.

## Omniglot protocol

Omniglot contains 1,623 character classes with 20 samples per class.

- The first 963 classes are used for meta-training.
- The remaining 660 mutually disjoint classes are used for meta-testing.
- Omniglot-1 treats every unseen character class as one sequential task.
- Per meta-test class, 15 samples are used for adaptation and five held-out
  samples are used for evaluation.
- Tasks arrive sequentially and are used for adaptation only once.
- Evaluation predicts over all classes observed so far.
- Only the final output layer is adapted during meta-testing.

Candidate fine-tuning learning rates are selected only on
background/meta-training classes. The selected rate is saved and frozen before
the 660 unseen meta-test tasks are evaluated. Meta-test query accuracy is not
used for learning-rate or checkpoint selection.

```bash
export OMNIGLOT_DATA_ROOT=/absolute/path/to/omni

# Preview all Table I commands.
python3 run_paper_five_runs.py \
  --dataset omniglot \
  --omniglot-root "$OMNIGLOT_DATA_ROOT"

# Train and evaluate both methods for all five seeds.
mkdir -p paper_runs/checkpoints
python3 run_paper_five_runs.py \
  --dataset omniglot \
  --omniglot-root "$OMNIGLOT_DATA_ROOT" \
  --execute
```

For every method and seed, the runner performs:

1. meta-training with `mrcl_classification.py`;
2. training-side selection with
   `evaluate_omniglot_continual.py --phase select`;
3. frozen testing with `evaluate_omniglot_continual.py --phase test`.

Outputs:

```text
paper_runs/selection/omniglot_<method>_seed<seed>_selection.json
paper_runs/results/omniglot/omniglot_<method>_seed<seed>_R_matrix.csv
paper_runs/results/omniglot/omniglot_<method>_seed<seed>_metrics.json
```

## CIFAR-100 protocols

CIFAR-100 contains 100 classes. The paper evaluates three task sizes:

- CIFAR100-5: five classes per task;
- CIFAR100-10: ten classes per task;
- CIFAR100-20: twenty classes per task.

For every seed, 60% of the class tasks are used for meta-training and the
remaining 40% mutually disjoint tasks are used for meta-testing. Fixed orders
are stored in `cifar100_protocols/`. The 5-, 10-, and 20-class settings for one
seed share the same underlying 100-class order.

```bash
# Generate or verify all 15 manifests.
python3 prepare_cifar100_protocol.py --seeds 9 19 29 39 49

export CIFAR100_DATA_ROOT=/absolute/path/to/cifar100

# Preview all Table II commands.
python3 run_paper_five_runs.py \
  --dataset cifar100 \
  --cifar-root "$CIFAR100_DATA_ROOT"

# Train and evaluate all task sizes, methods, and seeds.
mkdir -p paper_runs/checkpoints
python3 run_paper_five_runs.py \
  --dataset cifar100 \
  --cifar-root "$CIFAR100_DATA_ROOT" \
  --execute
```

Select a smaller subset when needed:

```bash
python3 run_paper_five_runs.py \
  --dataset cifar100 --method fseml --seeds 9 \
  --cifar-root "$CIFAR100_DATA_ROOT" --execute

python3 run_paper_five_runs.py \
  --dataset cifar100 --phase evaluate \
  --cifar-root "$CIFAR100_DATA_ROOT" --execute
```

`evaluate_cifar100_continual.py` writes:

```text
paper_runs/results/cifar100/cifar100_<N>_<method>_seed<seed>_R_matrix.csv
paper_runs/results/cifar100/cifar100_<N>_<method>_seed<seed>_metrics.json
```

## CORe50 protocol

The manuscript uses the official NICv2-79 temporal and batch structure. The 50
object classes are divided into 30 meta-training classes and 20 mutually
disjoint unseen meta-test classes. The 20 test classes form four consecutive
5-way tasks.

```bash
export CORE50_DATA_ROOT=/absolute/path/to/core50_128x128

bash setup_core50_metadata.sh
python3 verify_core50_data.py --data-root "$CORE50_DATA_ROOT"

```

See `CORE50_HELDOUT_RUNBOOK.md` for the full procedure. 
## CLINC150 protocol

CLINC150 contains 150 intents in ten application domains, with 15 intents per
domain. The manuscript protocol uses:

- six domains and 90 intents for meta-training;
- four mutually disjoint unseen domains and 60 intents for meta-testing;
- three consecutive 5-way tasks per meta-test domain, for 12 tasks in total;
- frozen 384-dimensional all-MiniLM-L6-v2 sentence embeddings.


## Metrics and result format

Each continual evaluator records an accuracy matrix `R`, where `R[i,j]` is the
test accuracy on task `j` after learning task `i`.

- **ACC**: mean final accuracy across all learned tasks;
- **FM**: mean decrease from each previous task's best accuracy to its final accuracy;
- **LA**: mean diagonal accuracy immediately after each task is learned.

Per-run JSON files store metrics as fractions in `[0,1]`. Summary files convert
them to percentage points and report the five-seed mean and sample standard
deviation.

## Package validation

```bash
# Run the minimum package tests.
python3 -m unittest discover -s tests -v

```

## Artifacts and provenance

`artifacts/` contains available historical logs, structured metrics,
configurations, replay analyses, and reconstruction-fidelity outputs. These
provide audit evidence but do not replace fresh reproduction runs.

- `artifacts/README.md` describes the artifact layout.
- `artifacts/PROVENANCE.md` records each imported artifact group.
- `artifacts/MANIFEST.sha256` records package hashes.

Dataset licenses and repository size make direct redistribution
inappropriate. 

## License and citation

Repository source code is released under the MIT License; see `LICENSE`.
<!-- Third-party material under `vendor/` retains its original licenses and
attribution requirements.

Citation metadata is provided in `CITATION.cff`. -->
