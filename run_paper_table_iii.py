#!/usr/bin/env python3
"""Run or print the Table III CORe50 and CLINC150 FSEML pipelines."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEEDS = (9, 19, 29, 39, 49)

def core50_commands(python, data_root, output_root, method, seed=9):
    output = output_root / f"core50_{method}"
    replay = ["--replay", "--replay-content", "decoder", "--replay-warmup", "100", "--buffer-size", "1000", "--replay-gap", "32", "--replay-rate", "1.0", "--replay-partitions", "50"] if method == "fseml-er" else ["--no-replay"]
    name = f"core50_heldout_fseml_er_decoder_seed{seed}" if method == "fseml-er" else f"core50_heldout_fseml_seed{seed}"
    checkpoint = output / f"{name}_best.pt"
    selection = output / f"{name}_validation_best.json"
    split = ROOT / "core50_heldout_splits" / f"split_seed{seed}.json"
    if not split.is_file(): raise FileNotFoundError(split)
    common_eval = ["--data-root", data_root, "--split", str(split), "--seed", str(seed), "--prediction-class-scope", "task"]
    return [
        [python, str(ROOT / "train_core50_heldout.py"), "--data-root", data_root, "--split", str(split), "--seed", str(seed), "--meta-iterations", "5000", "--ways", "5", "--support-shots", "2", "--query-shots", "5", "--learning-rate", "0.0001", "--inner-lr", "0.05", "--update-step", "10", "--validation-every", "1000", "--checkpoint-every", "1000", "--output-dir", str(output), *replay],
        [python, str(ROOT / "evaluate_core50_heldout.py"), "--checkpoint", str(checkpoint), *common_eval, "--phase", "validation", "--inner-lrs", "0.01", "0.05", "0.1", "--output", str(output / f"validation_seed{seed}.json")],
        [python, str(ROOT / "evaluate_core50_heldout.py"), "--checkpoint", str(checkpoint), *common_eval, "--phase", "test", "--selection", str(selection), "--output", str(output / f"test_seed{seed}.json")],
    ]

def clinc150_commands(python, features, output_root, method, seed=9):
    output = output_root / f"clinc150_{method}"
    replay = ["--replay", "--replay-warmup", "100", "--buffer-size", "1000", "--replay-gap", "32", "--replay-rate", "0.5", "--replay-loss-weight", "0.1"] if method == "fseml-er" else ["--no-replay"]
    name = f"clinc150_fseml_er_decoder_seed{seed}" if method == "fseml-er" else f"clinc150_fseml_seed{seed}"
    protocol = ROOT / "clinc150_protocols" / f"domain_holdout_6_4_seed{seed}.json"
    if not protocol.is_file(): raise FileNotFoundError(protocol)
    return [
        [python, str(ROOT / "train_clinc150_heldout.py"), "--features", features, "--protocol", str(protocol), "--seed", str(seed), "--meta-iterations", "30000", "--ways", "5", "--support-shots", "2", "--query-shots", "5", "--meta-lr", "0.0001", "--inner-lr", "0.01", "--update-step", "5", "--classifier-reset-init", "zero", "--hidden-dim", "256", "--latent-dim", "128", "--classifier-hidden", "512", "--validation-every", "2000", "--validation-online-lrs", "0.001", "0.005", "0.01", "0.02", "0.05", "--validation-adaptation-epochs", "1", "--validation-prediction-class-scope", "task", "--output-dir", str(output), *replay],
        [python, str(ROOT / "evaluate_clinc150_heldout.py"), "--checkpoint", str(output / f"{name}_best.pt"), "--selection", str(output / f"{name}_validation_best.json"), "--features", features, "--protocol", str(protocol), "--phase", "test", "--seed", str(seed), "--adaptation-epochs", "1", "--online-batch-size", "32", "--eval-batch-size", "256", "--output", str(output / f"{name}_test.json")],
    ]

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["all", "core50", "clinc150"], default="all")
    p.add_argument("--method", choices=["all", "fseml", "fseml-er"], default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--core50-root", default="../data/core50_128x128")
    p.add_argument("--clinc150-features", default="clinc150_features/clinc150_minilm_l6_v2.pt")
    p.add_argument("--output-root", type=Path, default=ROOT / "paper_runs" / "table_iii")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--execute", action="store_true", help="Execute commands; default is a safe dry run.")
    args = p.parse_args(argv)
    unknown = sorted(set(args.seeds) - set(SEEDS))
    if unknown: p.error(f"seeds not declared by the five-run protocol: {unknown}")
    methods = ["fseml", "fseml-er"] if args.method == "all" else [args.method]
    selected = []
    for seed in args.seeds:
        for method in methods:
            if args.dataset in ("all", "core50"): selected.extend(core50_commands(args.python, args.core50_root, args.output_root.resolve(), method, seed))
            if args.dataset in ("all", "clinc150"): selected.extend(clinc150_commands(args.python, args.clinc150_features, args.output_root.resolve(), method, seed))
    print(f"{'execute' if args.execute else 'dry-run'}: {len(selected)} commands")
    for index, command in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] {subprocess.list2cmdline(command)}", flush=True)
        if args.execute: subprocess.run(command, cwd=ROOT, check=True)
    return 0

if __name__ == "__main__": raise SystemExit(main())
