from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace

import torch

from datasets.core50_heldout import CORe50HeldOutData, load_split
from evaluate_core50_heldout import evaluate
from run_core50_experiment import set_seed


def checkpoint_step(path: Path) -> int:
    match = re.search(r"_step(\d+)\.pt$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse training step from {path}")
    return int(match.group(1))


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    split = load_split(args.split)
    classes = split["meta_validation_classes"]
    data = CORe50HeldOutData(
        args.data_root, args.metadata_root, run=args.run, image_size=args.image_size
    )
    eval_args = SimpleNamespace(
        ways=args.ways,
        support_shots=args.support_shots,
        adaptation_epochs=args.adaptation_steps,
        prediction_class_scope=args.prediction_class_scope,
        eval_batch_size=args.eval_batch_size,
        workers=args.workers,
        seed=args.seed,
        classifier_reset_init=args.classifier_reset_init,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob(args.pattern), key=checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints matched {args.pattern!r} in {checkpoint_dir}"
        )

    validations = []
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        candidates = [
            evaluate(checkpoint, data, classes, eval_args, inner_lr, device)
            for inner_lr in args.inner_lrs
        ]
        best = max(candidates, key=lambda item: item["final_ACC"])
        event = {
            "meta_iteration": checkpoint_step(path),
            "checkpoint": str(path),
            "candidates": candidates,
            "selected_inner_lr": best["inner_lr"],
            "validation_ACC": best["final_ACC"],
            "best": best,
        }
        validations.append(event)
        print(json.dumps({"validation": event}), flush=True)
        del checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selected = max(validations, key=lambda item: item["validation_ACC"])
    payload = {
        "phase": "validation_checkpoint_selection",
        "classes": classes,
        "support_shots": args.support_shots,
        "adaptation_steps": args.adaptation_steps,
        "classifier_reset_init": args.classifier_reset_init,
        "prediction_class_scope": args.prediction_class_scope,
        "inner_lrs": args.inner_lrs,
        "validations": validations,
        "selected_meta_iteration": selected["meta_iteration"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_inner_lr": selected["selected_inner_lr"],
        "best": selected["best"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"selection": {
        "selected_meta_iteration": payload["selected_meta_iteration"],
        "selected_checkpoint": payload["selected_checkpoint"],
        "selected_inner_lr": payload["selected_inner_lr"],
        "validation_ACC": payload["best"]["final_ACC"],
    }}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--pattern", default="*_step*.pt")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--metadata-root", default="vendor/core50_official")
    parser.add_argument("--split", default="core50_heldout_splits/split_seed9.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--ways", type=int, default=5)
    parser.add_argument("--support-shots", type=int, default=15)
    parser.add_argument("--adaptation-steps", type=int, default=10)
    parser.add_argument("--inner-lrs", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--prediction-class-scope", choices=["task", "seen"], default="task")
    parser.add_argument(
        "--classifier-reset-init",
        choices=["kaiming", "zero"],
        default="kaiming",
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
