#!/usr/bin/env bash
set -euo pipefail

temporary_dir="$(mktemp -d)"
trap 'rm -rf "$temporary_dir"' EXIT
git clone --depth 1 https://github.com/vlomonaco/core50.git "$temporary_dir/core50"
mkdir -p vendor/core50_official
cp "$temporary_dir/core50/LICENSE" vendor/core50_official/LICENSE
cp "$temporary_dir/core50/extras/paths.pkl" vendor/core50_official/paths.pkl
cp "$temporary_dir/core50/extras/LUP.pkl" vendor/core50_official/LUP.pkl
cp "$temporary_dir/core50/extras/labels.pkl" vendor/core50_official/labels.pkl
cp "$temporary_dir/core50/extras/batches_filelists_NICv2.zip" vendor/core50_official/batches_filelists_NICv2.zip
echo "CORe50 official metadata installed under vendor/core50_official"

expected_checksums="$(cat <<'EOF'
073e9a900fa558a31e9b9264b0cad7a3339c7b6d14ced0f602200ada1d2b694f  vendor/core50_official/paths.pkl
6aca5cd9af1ee9850bd0e6f25e865be2719a888faa2f708dd1142af16e5fa545  vendor/core50_official/LUP.pkl
123dcd893c3e4187d51712c1e7c1c809e4ddb06d56eb3aa4aaf42a7ad4f5c46c  vendor/core50_official/labels.pkl
197502ca31833d25612f46c8825e58fba4bfec55a4cf02c7b0db7f165c72d979  vendor/core50_official/batches_filelists_NICv2.zip
EOF
)"
printf '%s\n' "$expected_checksums" | shasum -a 256 -c -

if [[ -n "${CORE50_DATA_ROOT:-}" ]]; then
  "${PYTHON_BIN:-python}" verify_core50_data.py --data-root "$CORE50_DATA_ROOT"
else
  echo "Set CORE50_DATA_ROOT and run verify_core50_data.py to verify the extracted images."
fi
