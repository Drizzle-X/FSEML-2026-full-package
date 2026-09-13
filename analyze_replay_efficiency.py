from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict


EVALUATION_KEYS = (
    "classes_per_task",
    "seed",
    "task_protocol",
    "evaluation_mode",
    "finetune_scope",
    "finetune_lr",
    "srn_finetune_lr",
    "finetune_epochs",
    "finetune_batch_size",
    "finetune_order",
    "loss_class_scope",
    "prediction_class_scope",
    "optimizer",
    "reset_output",
    "data_augmentation",
)


def load_metrics(path: str) -> Dict:
    with open(path) as handle:
        payload = json.load(handle)
    if "final" not in payload or "ACC" not in payload["final"]:
        raise ValueError(f"Metrics file has no final ACC: {path}")
    return payload


def normalized_value(key: str, value):
    if key == "task_protocol" and isinstance(value, str):
        return os.path.basename(value)
    return value


def validate_protocol(reference: Dict, candidate: Dict, candidate_name: str) -> None:
    mismatches = []
    for key in EVALUATION_KEYS:
        left = normalized_value(key, reference.get(key))
        right = normalized_value(key, candidate.get(key))
        if left != right:
            mismatches.append(f"{key}: no_replay={left!r}, {candidate_name}={right!r}")
    if mismatches:
        raise ValueError(
            f"Evaluation protocol mismatch for {candidate_name}:\n  "
            + "\n  ".join(mismatches)
        )


def main(args) -> None:
    no_replay = load_metrics(args.no_replay)
    decoder = load_metrics(args.decoder_replay)
    raw = load_metrics(args.raw_replay)
    validate_protocol(no_replay, decoder, "decoder_replay")
    validate_protocol(no_replay, raw, "raw_replay")

    acc_none = float(no_replay["final"]["ACC"])
    acc_decoder = float(decoder["final"]["ACC"])
    acc_raw = float(raw["final"]["ACC"])
    decoder_gain = acc_decoder - acc_none
    raw_gain = acc_raw - acc_none
    efficiency = decoder_gain / raw_gain if not math.isclose(raw_gain, 0.0) else float("nan")

    result = {
        "dataset": f"CIFAR100-{int(no_replay['classes_per_task'])}",
        "seed": int(no_replay["seed"]),
        "acc_no_replay": acc_none,
        "acc_decoder_replay": acc_decoder,
        "acc_raw_replay": acc_raw,
        "decoder_gain": decoder_gain,
        "raw_gain": raw_gain,
        "replay_efficiency": efficiency,
        "replay_efficiency_percent": efficiency * 100.0,
        "metrics_files": {
            "no_replay": os.path.abspath(args.no_replay),
            "decoder_replay": os.path.abspath(args.decoder_replay),
            "raw_replay": os.path.abspath(args.raw_replay),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    with open(args.output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[key for key in result if key != "metrics_files"],
        )
        writer.writeheader()
        writer.writerow({key: value for key, value in result.items() if key != "metrics_files"})
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute decoder replay gain and efficiency relative to matched raw replay."
    )
    parser.add_argument("--no-replay", required=True)
    parser.add_argument("--decoder-replay", required=True)
    parser.add_argument("--raw-replay", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    main(parser.parse_args())
