#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
FEATURES=clinc150_features/clinc150_minilm_l6_v2.pt
PROTOCOL=clinc150_protocols/domain_holdout_6_4_seed9.json

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi
echo "Using Python: $(command -v "$PYTHON_BIN")"

MPLCONFIGDIR=/tmp/matplotlib-clinc-fseml "$PYTHON_BIN" -u train_clinc150_heldout.py \
  --features "$FEATURES" --protocol "$PROTOCOL" --seed 9 \
  --meta-iterations 30000 --ways 5 --support-shots 2 --query-shots 5 \
  --meta-lr 0.0001 --inner-lr 0.01 --update-step 5 --meta-batch-size 1 \
  --classifier-reset-init zero --hidden-dim 256 --latent-dim 128 --classifier-hidden 512 \
  --log-every 200 --checkpoint-every 2000 --validation-every 2000 \
  --validation-online-lrs 0.001 0.005 0.01 0.02 0.05 \
  --validation-adaptation-epochs 1 --validation-prediction-class-scope task \
  --online-batch-size 32 --eval-batch-size 256 \
  --workers 4 --no-replay --output-dir clinc150_heldout_fseml_30k

MPLCONFIGDIR=/tmp/matplotlib-clinc-er "$PYTHON_BIN" -u train_clinc150_heldout.py \
  --features "$FEATURES" --protocol "$PROTOCOL" --seed 9 \
  --meta-iterations 30000 --ways 5 --support-shots 2 --query-shots 5 \
  --meta-lr 0.0001 --inner-lr 0.01 --update-step 5 --meta-batch-size 1 \
  --classifier-reset-init zero --hidden-dim 256 --latent-dim 128 --classifier-hidden 512 \
  --log-every 200 --checkpoint-every 2000 --validation-every 2000 \
  --validation-online-lrs 0.001 0.005 0.01 0.02 0.05 \
  --validation-adaptation-epochs 1 --validation-prediction-class-scope task \
  --online-batch-size 32 --eval-batch-size 256 \
  --workers 4 --replay --replay-warmup 100 --buffer-size 1000 --replay-gap 32 --replay-rate 0.5 \
  --replay-loss-weight 0.1 \
  --output-dir clinc150_heldout_fseml_er_30k
