#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
DATA_FILE=vendor/oos-eval/data/data_full.json
DOMAINS_FILE=vendor/oos-eval/data/domains.json
FEATURE_FILE=clinc150_features/clinc150_minilm_l6_v2.pt
PROTOCOL_FILE=clinc150_protocols/domain_holdout_6_4_seed9.json

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi
echo "Using Python: $(command -v "$PYTHON_BIN")"
"$PYTHON_BIN" -c 'import sys; print("Python version:", sys.version); print("Environment prefix:", sys.prefix)'
"$PYTHON_BIN" - <<'PY'
import sys
import torch

if sys.version_info < (3, 10):
    raise SystemExit(f"CLINC150 dependencies require Python >= 3.10, found {sys.version}")
parts = torch.__version__.split("+", 1)[0].split(".")
torch_version = tuple(int("".join(ch for ch in part if ch.isdigit()) or 0) for part in parts[:2])
if torch_version < (2, 2):
    raise SystemExit(
        f"sentence-transformers 6.0.0 requires torch >= 2.2; found {torch.__version__}. "
        "Upgrade the CUDA-enabled conda environment before running pip."
    )
print("Existing CUDA PyTorch preserved:", torch.__version__)
PY

"$PYTHON_BIN" -m pip install -r requirements_clinc150.txt

"$PYTHON_BIN" - <<'PY'
from importlib.metadata import version

for package in ("datasets", "sentence-transformers", "transformers", "pyarrow", "huggingface-hub"):
    print(f"{package}:", version(package))
PY

if [[ ! -f "$DATA_FILE" || ! -f "$DOMAINS_FILE" ]]; then
  mkdir -p vendor
  if [[ -d vendor/oos-eval/.git ]]; then
    git -C vendor/oos-eval pull --ff-only || true
  else
    git clone --depth 1 https://github.com/clinc/oos-eval.git vendor/oos-eval || true
  fi
  if [[ ! -f "$DATA_FILE" || ! -f "$DOMAINS_FILE" ]]; then
    "$PYTHON_BIN" download_clinc150_hf.py --output-dir vendor/oos-eval/data
  fi
fi

if [[ ! -f "$FEATURE_FILE" ]]; then
  "$PYTHON_BIN" prepare_clinc150_embeddings.py \
    --data "$DATA_FILE" \
    --encoder sentence-transformers/all-MiniLM-L6-v2 \
    --output "$FEATURE_FILE"
fi

"$PYTHON_BIN" prepare_clinc150_protocol.py \
  --domains "$DOMAINS_FILE" \
  --seed 9 \
  --output "$PROTOCOL_FILE"

"$PYTHON_BIN" - <<'PY'
from datasets.clinc150_continual import CLINCProtocol, load_feature_payload

features = load_feature_payload("clinc150_features/clinc150_minilm_l6_v2.pt")
protocol = CLINCProtocol.load("clinc150_protocols/domain_holdout_6_4_seed9.json")
assert features["embeddings"].shape[0] == 23700
assert features["embeddings"].shape[1] == 384
assert len(protocol.meta_train_domains) == 6
assert len(protocol.meta_validation_domains) == 0
assert len(protocol.meta_test_domains) == 4
assert len(protocol.validation_tasks) == 18
assert len(protocol.test_tasks) == 12
print("CLINC150 server preparation OK")
PY
