# CLINC150 6/4-domain paper protocol

## Protocol

This is a feature-level cross-modal validation of FSEML, not end-to-end
continual language-model training:

```text
utterance -> frozen all-MiniLM-L6-v2 -> 384-D embedding -> vector FSSAE -> CPN
```

The official 10 intent domains are partitioned before training:

- meta-train: 6 domains / 90 intents;
- meta-test: 4 mutually disjoint domains / 60 intents;
- each of the four meta-test domains forms three consecutive 5-way tasks,
  producing 12 meta-test tasks in total.

Every task introduces five intents. Meta-training and meta-test domain and
intent sets are mutually disjoint. The primary protocol follows the task-aware held-out
adaptation setting used by the project's Omniglot, CIFAR100 and CORe50
experiments: prediction is restricted to the five intents of the task being
evaluated. A stricter seen-class protocol is retained as an optional diagnostic
through `--validation-prediction-class-scope seen`. Official `train`
utterances form the adaptation stream, official `validation` utterances are
used only for checkpoint/LR selection, and official `test` utterances are
accessed only by the final evaluation command.

## Environment and server-only data preparation

All commands below assume that the current working directory is the root of
this cloned repository.

Create and activate the environment declared in `environment.yml`. The server
scripts use `python` from the active environment and print its resolved path
and environment prefix before doing any work. An explicit interpreter can also
be provided with `PYTHON_BIN=/absolute/path/to/python`.

All dependencies are contained in this repository. `requirements.txt` pins the
validated shared environment and includes the CLINC150-specific dependencies
from `requirements_clinc150.txt`.

The following command first attempts the official CLINC GitHub repository. If
GitHub is unavailable, it automatically falls back to the complete 23,700-row
`contemmcm/clinc150` Hugging Face mirror and reconstructs the official JSON
files on the server. It also downloads the frozen sentence encoder on the
server only. It does not need any dataset copied from the local machine:

```bash
bash prepare_clinc150_server.sh
```

If the server requires the Hugging Face mirror endpoint:

```bash
HF_ENDPOINT=https://hf-mirror.com bash prepare_clinc150_server.sh
```

Expected checks:

```text
23700 x 384 embeddings
6 meta-train domains / 90 intents
4 meta-test domains / 60 intents
12 reported sequential 5-way tasks
CLINC150 server preparation OK
```

Hash embeddings are smoke-test artifacts and are rejected by the training and
evaluation entry points.

## Short smoke runs

FSEML:

```bash
MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-clinc-smoke" python -u train_clinc150_heldout.py --features clinc150_features/clinc150_minilm_l6_v2.pt --protocol clinc150_protocols/domain_holdout_6_4_seed9.json --seed 9 --meta-iterations 10 --ways 5 --support-shots 2 --query-shots 5 --update-step 1 --log-every 1 --validation-every 10 --validation-online-lrs 0.005 0.01 --validation-adaptation-epochs 1 --validation-prediction-class-scope task --workers 0 --no-replay --output-dir clinc150_heldout_smoke
```

FSEML-ER decoder audit:

```bash
MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-clinc-er-smoke" python -u train_clinc150_heldout.py --features clinc150_features/clinc150_minilm_l6_v2.pt --protocol clinc150_protocols/domain_holdout_6_4_seed9.json --seed 9 --meta-iterations 110 --ways 5 --support-shots 2 --query-shots 5 --update-step 1 --log-every 10 --validation-every 110 --validation-online-lrs 0.005 0.01 --validation-adaptation-epochs 1 --validation-prediction-class-scope task --workers 0 --replay --replay-warmup 10 --buffer-size 1000 --replay-gap 32 --replay-rate 0.5 --replay-loss-weight 0.1 --output-dir clinc150_heldout_er_smoke
```

The ER log must show non-zero `replay_events`, `total_replayed_samples`, and
balanced per-label occupancy.

## Full meta-training and validation

`run_clinc150_server.sh` runs the two methods sequentially. For parallel GPU
runs, use the equivalent one-line `nohup` commands recorded with the results.

```bash
bash run_clinc150_server.sh
```

Primary configuration:

- seed 9;
- 30,000 outer updates;
- 5-way, 2 support and 5 query utterances per intent;
- frozen sentence encoder;
- vector FSSAE dimensions 384 -> 256 -> 128;
- CPN hidden dimension 512;
- zero-reset episodic classifier;
- validation every 2,000 updates;
- online LR selected from `0.001, 0.005, 0.01, 0.02, 0.05`;
- one single-pass adaptation epoch;
- task-aware five-intent prediction for the primary held-out protocol;
- FSEML-ER stores 128-D latent vectors and reconstructs 384-D pseudo-embeddings;
- final test adaptation is replay-free for both methods.

## Final meta-test

Do not run these commands until meta-training is finished and the corresponding
`validation_best.json` exists. Do not modify the model or hyperparameters after
reading test results.

FSEML:

```bash
MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-clinc-test" python -u evaluate_clinc150_heldout.py --checkpoint clinc150_heldout_fseml_30k/clinc150_fseml_seed9_best.pt --selection clinc150_heldout_fseml_30k/clinc150_fseml_seed9_validation_best.json --features clinc150_features/clinc150_minilm_l6_v2.pt --protocol clinc150_protocols/domain_holdout_6_4_seed9.json --phase test --seed 9 --adaptation-epochs 1 --online-batch-size 32 --eval-batch-size 256 --workers 4 --output clinc150_heldout_fseml_30k/clinc150_fseml_seed9_test.json
```

FSEML-ER:

```bash
MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-clinc-er-test" python -u evaluate_clinc150_heldout.py --checkpoint clinc150_heldout_fseml_er_30k/clinc150_fseml_er_decoder_seed9_best.pt --selection clinc150_heldout_fseml_er_30k/clinc150_fseml_er_decoder_seed9_validation_best.json --features clinc150_features/clinc150_minilm_l6_v2.pt --protocol clinc150_protocols/domain_holdout_6_4_seed9.json --phase test --seed 9 --adaptation-epochs 1 --online-batch-size 32 --eval-batch-size 256 --workers 4 --output clinc150_heldout_fseml_er_30k/clinc150_fseml_er_decoder_seed9_test.json
```

Report the twelve-task accuracy matrix, Final ACC, Learning ACC and Forgetting.
Validation results must not be labeled as test results.
