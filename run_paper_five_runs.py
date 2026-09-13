#!/usr/bin/env python3
"""Run or print the five-seed FSEML/FSEML-ER main experiment commands."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "paper_main_five_runs.json"

def commands(config, python, datasets, methods, seeds, cifar_root, omniglot_root, output_root):
    shared = config["shared_training"]
    replay = config["replay"]
    common = [
        "--steps", str(shared["meta_updates"]), "--meta_lr", str(shared["meta_learning_rate"]),
        "--update_lr", str(shared["inner_learning_rate"]), "--update_step", str(shared["inner_steps"]),
        "--weight-decay-weight", str(shared["weight_decay"]),
        "--reconstruction-weight", str(shared["reconstruction_weight"]),
        "--fisher-reconstruction-weight", str(shared["fisher_reconstruction_weight"]),
        "--sparsity-target", str(shared["sparsity_target"]), "--sparsity-weight", str(shared["sparsity_weight"]),
        "--fda-weight", str(shared["fda_weight"]), "--encoder-channels", str(shared["encoder_channels"]),
        "--latent-dim", str(shared["latent_dim"]), "--adapter-dim", str(shared["adapter_dim"]),
        "--tasks", "1", "--no-reset",
    ]
    replay_args = [
        "--replay", "--replay-mode", replay["mode"], "--replay-content", replay["content"],
        "--replay-buffer-size", str(replay["buffer_size"]), "--replay-gap", str(replay["gap"]),
        "--replay-rate", str(replay["rate"]), "--replay-top-p", str(replay["top_p"]),
        "--replay-partitions", str(replay["partitions"]),
    ]
    for seed in seeds:
        for method in methods:
            method_args = replay_args if method == "fseml-er" else ["--no-replay"]
            if "cifar100" in datasets:
                c = config["cifar100"]
                for size in c["classes_per_task"]:
                    protocol = ROOT / "cifar100_protocols" / f"cifar100_{size}classes_seed{seed}.json"
                    if not protocol.is_file(): raise FileNotFoundError(protocol)
                    stem = f"cifar100_{size}_{method}_seed{seed}"
                    checkpoint = output_root / "checkpoints" / f"{stem}.net"
                    yield "train", [python, str(ROOT / "mrcl_classification.py"), "--dataset", "CIFAR100", "--data-root", cifar_root,
                        "--classes-per-task", str(size), "--task-protocol", str(protocol), "--seed", str(seed),
                        "--query-scope", c["query_scope"], "--support-sampling", c["support_sampling"],
                        "--name", stem, "--model_name", str(checkpoint), *common, *method_args]
                    e = c["evaluation"]
                    yield "evaluate", [python, str(ROOT / "evaluate_cifar100_continual.py"), "--model", str(checkpoint),
                        "--data-root", cifar_root, "--task-protocol", str(protocol), "--classes-per-task", str(size),
                        "--seed", str(seed), "--evaluation-mode", e["mode"], "--finetune-lr", str(e["finetune_learning_rate"]),
                        "--finetune-epochs", str(e["finetune_epochs"]), "--finetune-batch-size", str(e["finetune_batch_size"]),
                        "--finetune-scope", e["finetune_scope"], "--loss-class-scope", e["loss_class_scope"],
                        "--prediction-class-scope", e["prediction_class_scope"], "--run-name", stem,
                        "--output-dir", str(output_root / "results" / "cifar100")]
            if "omniglot" in datasets:
                o = config["omniglot"]
                stem = f"omniglot_{method}_seed{seed}"
                checkpoint = output_root / "checkpoints" / f"{stem}.net"
                selection = output_root / "selection" / f"{stem}_selection.json"
                yield "train", [python, str(ROOT / "mrcl_classification.py"), "--dataset", "omniglot", "--seed", str(seed),
                    "--name", stem, "--model_name", str(checkpoint), *common, *method_args]
                yield "select", [python, str(ROOT / "evaluate_omniglot_continual.py"), "--phase", "select",
                    "--dataset-path", omniglot_root, "--model", str(checkpoint), "--seed", str(seed),
                    "--epochs", str(o["finetune_epochs"]), "--selection-tasks", str(o["selection_tasks"]),
                    "--lr-candidates", *[str(value) for value in o["learning_rate_candidates"]],
                    "--output", str(selection)]
                yield "evaluate", [python, str(ROOT / "evaluate_omniglot_continual.py"), "--phase", "test",
                    "--dataset-path", omniglot_root, "--model", str(checkpoint), "--seed", str(seed),
                    "--epochs", str(o["finetune_epochs"]), "--selection", str(selection), "--run-name", stem,
                    "--output", str(output_root / "results" / "omniglot")]

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--dataset", choices=["all", "cifar100", "omniglot"], default="all")
    p.add_argument("--method", choices=["all", "fseml", "fseml-er"], default="all")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--phase", choices=["all", "train", "select", "evaluate"], default="all")
    p.add_argument("--cifar-root", default="../data/cifar100")
    p.add_argument("--omniglot-root", default="../data/omni")
    p.add_argument("--output-root", type=Path, default=ROOT / "paper_runs")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--execute", action="store_true", help="Execute commands; default is a safe dry run.")
    args = p.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selected_seeds = args.seeds or config["seeds"]
    unknown = sorted(set(selected_seeds) - set(config["seeds"]))
    if unknown: p.error(f"seeds not declared by the five-run config: {unknown}")
    datasets = ["cifar100", "omniglot"] if args.dataset == "all" else [args.dataset]
    methods = ["fseml", "fseml-er"] if args.method == "all" else [args.method]
    selected = [(phase, command) for phase, command in commands(config, args.python, datasets, methods, selected_seeds,
        args.cifar_root, args.omniglot_root, args.output_root.resolve()) if args.phase == "all" or args.phase == phase]
    print(json.dumps({"mode": "execute" if args.execute else "dry-run", "commands": len(selected), "seeds": selected_seeds}))
    for index, (phase, command) in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] {phase}: " + subprocess.list2cmdline(command), flush=True)
        if args.execute: subprocess.run(command, cwd=ROOT, check=True)
    return 0

if __name__ == "__main__": raise SystemExit(main())
