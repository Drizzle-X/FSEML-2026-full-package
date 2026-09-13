from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from clinc150_heldout_common import model_from_checkpoint, run_continual_evaluation, set_seed
from datasets.clinc150_continual import CLINCProtocol, load_feature_payload


def main(args):
    if args.phase != "test":
        raise ValueError("This entry point is intentionally restricted to the final held-out test phase")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    protocol = CLINCProtocol.load(args.protocol)
    feature_payload = load_feature_payload(args.features)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(checkpoint, device)
    selection = json.loads(Path(args.selection).read_text())
    if int(selection["selected_meta_iteration"]) != int(checkpoint["meta_iteration"]):
        raise ValueError("Selection manifest and checkpoint refer to different meta-training iterations")
    trained_args = checkpoint["arguments"]
    expected_epochs = int(trained_args["validation_adaptation_epochs"])
    expected_batch_size = int(trained_args["online_batch_size"])
    if args.adaptation_epochs != expected_epochs or args.online_batch_size != expected_batch_size:
        raise ValueError(
            "Test adaptation must match the validation protocol: "
            f"epochs={expected_epochs}, batch_size={expected_batch_size}"
        )
    online_lr = float(selection["selected_online_lr"])
    result = run_continual_evaluation(
        model,
        feature_path=args.features,
        feature_payload=feature_payload,
        protocol=protocol,
        phase="test",
        online_lr=online_lr,
        adaptation_epochs=args.adaptation_epochs,
        batch_size=args.online_batch_size,
        eval_batch_size=args.eval_batch_size,
        workers=args.workers,
        seed=args.seed + 5000,
        device=device,
        classifier_reset_init=checkpoint["arguments"].get("classifier_reset_init", "zero"),
        prediction_class_scope=trained_args.get("validation_prediction_class_scope", "task"),
    )
    payload = {
        **result,
        "checkpoint": args.checkpoint,
        "checkpoint_meta_iteration": checkpoint["meta_iteration"],
        "selection": args.selection,
        "encoder": feature_payload["encoder"],
        "prediction_class_scope": trained_args.get("validation_prediction_class_scope", "task"),
        "adaptation_replay": False,
        "arguments": vars(args),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({key: payload[key] for key in ("final_ACC", "learning_ACC", "forgetting")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--features", default="clinc150_features/clinc150_minilm_l6_v2.pt")
    parser.add_argument("--protocol", default="clinc150_protocols/domain_holdout_6_4_seed9.json")
    parser.add_argument("--phase", choices=["test"], default="test")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--adaptation-epochs", type=int, default=1)
    parser.add_argument("--online-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
