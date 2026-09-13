from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.core50_heldout import CORe50HeldOutData, load_split
from model.meta_learner_fseml import MetaLearnerFSEML
from run_core50_experiment import learner_args, replay_checkpoint, set_seed
from evaluate_core50_heldout import evaluate as evaluate_heldout


def one_batch(dataset, workers):
    return next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=workers)))


def preserve_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def validation_args(args):
    return SimpleNamespace(
        ways=args.ways,
        support_shots=args.validation_support_shots,
        adaptation_epochs=args.validation_adaptation_epochs,
        prediction_class_scope=args.validation_prediction_class_scope,
        eval_batch_size=args.eval_batch_size,
        workers=args.workers,
        seed=args.seed,
        classifier_reset_init=args.classifier_reset_init,
    )


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    split = load_split(args.split)
    classes = split["meta_train_classes"]
    args.replay_class_ids = classes
    data = CORe50HeldOutData(
        args.data_root, args.metadata_root, run=args.run, image_size=args.image_size
    )
    learner = MetaLearnerFSEML(learner_args(args)).to(device)
    rng = random.Random(args.seed)
    events = []
    validation_events = []
    best_validation_acc = -1.0
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    method = "fseml_er_decoder" if args.replay else "fseml"
    name = f"core50_heldout_{method}_seed{args.seed}"

    def checkpoint_payload(iteration):
        return {
            "model": copy.deepcopy(learner.net).cpu(),
            "replay": replay_checkpoint(learner.replay),
            "optimizer_state": learner.optimizer.state_dict(),
            "meta_iteration": iteration,
            "split": split,
            "arguments": vars(args),
        }

    for iteration in range(args.meta_iterations):
        if hasattr(learner.replay, "sampling_enabled"):
            learner.replay.sampling_enabled = iteration >= args.replay_warmup
        episode_classes_batch = []
        accuracy_sum = loss_sum = fssae_loss_sum = classification_loss_sum = 0.0
        for meta_batch_index in range(args.meta_batch_size):
            episode_classes = sorted(rng.sample(classes, args.ways))
            episode_classes_batch.append(episode_classes)
            learner.set_active_classes(episode_classes)


            if args.episodic_reset:
                for label in episode_classes:
                    learner.reset_classifer(
                        label,
                        initialization=args.classifier_reset_init,
                        clear_optimizer_state=True,
                    )
            support, query = data.episode(
                episode_classes, args.support_shots, args.query_shots, rng
            )
            support_x, support_y = one_batch(support, args.workers)
            query_x, query_y = one_batch(query, args.workers)
            learner.train()
            accuracy, loss, fssae_loss, classification_loss = learner(

                support_x.unsqueeze(0).expand(args.update_step, *support_x.shape).to(device),
                support_y.unsqueeze(0).expand(args.update_step, *support_y.shape).to(device),
                query_x.unsqueeze(0).to(device),
                query_y.unsqueeze(0).to(device),
                zero_meta_grad=meta_batch_index == 0,
                step_meta_optimizer=meta_batch_index == args.meta_batch_size - 1,
                meta_loss_scale=1.0 / args.meta_batch_size,
            )
            accuracy_sum += float(accuracy)
            loss_sum += float(loss.detach())
            fssae_loss_sum += float(fssae_loss.detach())
            classification_loss_sum += float(classification_loss.detach())

        accuracy = accuracy_sum / args.meta_batch_size
        loss_value = loss_sum / args.meta_batch_size
        fssae_loss_value = fssae_loss_sum / args.meta_batch_size
        classification_loss_value = classification_loss_sum / args.meta_batch_size
        if iteration == 0 or (iteration + 1) % args.log_every == 0 or iteration + 1 == args.meta_iterations:
            event = {
                "meta_iteration": iteration + 1,
                "episode_classes": episode_classes_batch[-1],
                "meta_batch_episode_classes": episode_classes_batch,
                "meta_batch_size": args.meta_batch_size,
                "query_ACC": float(accuracy),
                "loss": loss_value,
                "classification_loss": classification_loss_value,
                "fssae_loss": fssae_loss_value,
                "replay": learner.replay.status(),
            }
            events.append(event)
            print(json.dumps(event), flush=True)

        completed = iteration + 1
        if args.checkpoint_every > 0 and completed % args.checkpoint_every == 0:
            torch.save(checkpoint_payload(completed), output / f"{name}_step{completed}.pt")

        should_validate = (
            args.validation_every > 0
            and (completed % args.validation_every == 0 or completed == args.meta_iterations)
        )
        if should_validate:
            rng_state = preserve_rng_state()
            checkpoint_view = {"model": learner.net}
            candidates = [
                evaluate_heldout(
                    checkpoint_view,
                    data,
                    split["meta_validation_classes"],
                    validation_args(args),
                    inner_lr,
                    device,
                )
                for inner_lr in args.validation_inner_lrs
            ]
            restore_rng_state(rng_state)
            best_at_step = max(candidates, key=lambda item: item["final_ACC"])
            validation_event = {
                "meta_iteration": completed,
                "candidates": candidates,
                "selected_inner_lr": best_at_step["inner_lr"],
                "validation_ACC": best_at_step["final_ACC"],
            }
            validation_events.append(validation_event)
            print(json.dumps({"validation": validation_event}), flush=True)
            if best_at_step["final_ACC"] > best_validation_acc:
                best_validation_acc = best_at_step["final_ACC"]
                best_payload = checkpoint_payload(completed)
                best_payload["validation"] = validation_event
                torch.save(best_payload, output / f"{name}_best.pt")
                (output / f"{name}_validation_best.json").write_text(
                    json.dumps({
                        "phase": "validation",
                        "selected_meta_iteration": completed,
                        "selected_inner_lr": best_at_step["inner_lr"],
                        "best": best_at_step,
                    }, indent=2)
                )

    checkpoint = checkpoint_payload(args.meta_iterations)
    torch.save(checkpoint, output / f"{name}.pt")
    (output / f"{name}_metrics.json").write_text(
        json.dumps({
            "run": name,
            "events": events,
            "validation_events": validation_events,
            "best_validation_ACC": best_validation_acc,
            "arguments": vars(args),
        }, indent=2)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--metadata-root", default="vendor/core50_official")
    p.add_argument("--split", default="core50_heldout_splits/split_seed9.json")
    p.add_argument("--run", type=int, default=0)
    p.add_argument("--seed", type=int, default=9)
    p.add_argument("--output-dir", default="core50_heldout_models")
    p.add_argument("--meta-iterations", type=int, default=5000)
    p.add_argument("--ways", type=int, default=5)
    p.add_argument("--support-shots", type=int, default=2)
    p.add_argument("--query-shots", type=int, default=5)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--validation-every", type=int, default=1000)
    p.add_argument("--validation-inner-lrs", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    p.add_argument("--validation-support-shots", type=int, default=2)
    p.add_argument(
        "--validation-adaptation-epochs",
        type=int,
        default=10,
        help="Full-support inner-loop steps used by held-out validation.",
    )
    p.add_argument("--validation-prediction-class-scope", choices=["task", "seen"], default="task")
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--replay-warmup", type=int, default=100)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--encoder-channels", type=int, default=16)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--adapter-dim", type=int, default=256)
    p.add_argument("--cpn-hidden-dim", type=int, default=2304)
    p.add_argument("--normalization", choices=["batch", "group"], default="batch")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--inner-lr", type=float, default=0.05)
    p.add_argument("--update-step", type=int, default=10)
    p.add_argument("--meta-batch-size", type=int, default=1)
    p.add_argument(
        "--episodic-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset the current episode classifier rows before support adaptation.",
    )
    p.add_argument(
        "--classifier-reset-init",
        choices=["kaiming", "zero"],
        default="kaiming",
        help="Classifier-row initialization shared by meta-train and held-out adaptation.",
    )
    p.add_argument("--query-batch-size", type=int, default=25)
    p.add_argument("--replay", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--replay-content", choices=["decoder"], default="decoder")
    p.add_argument("--buffer-size", type=int, default=1000)
    p.add_argument("--replay-gap", type=int, default=32)
    p.add_argument("--replay-rate", type=float, default=1.0)
    p.add_argument("--replay-partitions", type=int, default=50)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    main(p.parse_args())
