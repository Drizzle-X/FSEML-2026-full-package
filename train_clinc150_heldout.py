from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from clinc150_heldout_common import (
    ClassBalancedLatentReplay,
    classifier_accuracy,
    masked_loss,
    model_config_from_args,
    select_validation_candidate,
    set_seed,
)
from datasets.clinc150_continual import CLINCProtocol, EmbeddingDataset, load_feature_payload
from model.vector_fseml import VectorFSEML


def save_checkpoint(path, model, optimizer, config, args, meta_iteration, best_validation_acc):
    torch.save(
        {
            "format_version": 2,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": asdict(config),
            "arguments": vars(args),
            "meta_iteration": meta_iteration,
            "best_validation_ACC": best_validation_acc,
        },
        path,
    )


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocol = CLINCProtocol.load(args.protocol)
    feature_payload = load_feature_payload(args.features)
    if feature_payload["intent_to_id"] != protocol.intent_to_id:
        raise ValueError("Protocol and embedding files use different intent-to-ID mappings")

    train_intents = sorted(
        intent for intent, domain in protocol.intent_to_domain.items() if domain in protocol.meta_train_domains
    )
    train_data = EmbeddingDataset(args.features, "train", train_intents, payload=feature_payload)
    by_label = {int(label): torch.where(train_data.y == label)[0].tolist() for label in train_data.y.unique()}
    labels = sorted(by_label)
    required_per_class = args.support_shots + args.query_shots
    if any(len(indices) < required_per_class for indices in by_label.values()):
        raise ValueError("Not enough train utterances for the requested support/query episode")

    config = model_config_from_args(args, input_dim=feature_payload["embeddings"].shape[1])
    model = VectorFSEML(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr)
    episode_rng = np.random.RandomState(args.seed)
    replay_rng = random.Random(args.seed + 3000)
    replay = ClassBalancedLatentReplay(args.buffer_size, labels)
    events, validation_events = [], []
    best_validation_acc = -math.inf
    run_name = f"clinc150_{'fseml_er_decoder' if args.replay else 'fseml'}_seed{args.seed}"

    for meta_iteration in range(1, args.meta_iterations + 1):
        optimizer.zero_grad()
        batch_stats = []
        for _ in range(args.meta_batch_size):
            episode = episode_rng.choice(labels, args.ways, replace=False).tolist()
            model.reset_classifier_rows(episode, optimizer=optimizer, initialization=args.classifier_reset_init)
            support_idx, query_idx = [], []
            for label in episode:
                selected = episode_rng.choice(by_label[label], required_per_class, replace=False)
                support_idx.extend(selected[: args.support_shots])
                query_idx.extend(selected[args.support_shots :])
            sx, sy = train_data.x[support_idx].to(device), train_data.y[support_idx].to(device)
            qx, qy = train_data.x[query_idx].to(device), train_data.y[query_idx].to(device)

            fast_weights = model.cpn_parameters()
            for _ in range(args.update_step):
                support_out = model(sx, fast_weights)
                support_loss = masked_loss(support_out["logits"], sy, episode)
                gradients = torch.autograd.grad(support_loss, fast_weights, create_graph=True)
                fast_weights = [weight - args.inner_lr * gradient for weight, gradient in zip(fast_weights, gradients)]

            replay_x = replay_y = None
            if args.replay and meta_iteration > args.replay_warmup and replay.ready(args.replay_gap, args.replay_warmup):
                replay_count = min(len(replay), max(1, int(len(qx) * args.replay_rate)))
                sample = replay.sample(replay_count, replay_rng)
                if sample is not None:
                    replay_z, replay_y = sample
                    replay_x = model.srn.decode(replay_z.to(device)).detach()
                    replay_y = replay_y.to(device)
                    replay.consume_gap(args.replay_gap)

            query_out = model(qx, fast_weights)
            classification_loss = masked_loss(query_out["logits"], qy, episode)
            fssae = model.srn.loss(qx, query_out["latent"], query_out["reconstruction"], qy)
            loss = classification_loss + fssae["total"]
            replay_loss_value = 0.0
            if replay_x is not None:
                replay_out = model(replay_x)
                replay_classes = sorted(set(replay_y.detach().cpu().tolist()))
                replay_classification = masked_loss(replay_out["logits"], replay_y, replay_classes)
                replay_fssae = model.srn.loss(
                    replay_x, replay_out["latent"], replay_out["reconstruction"], replay_y
                )["total"]
                replay_auxiliary = replay_classification + replay_fssae
                loss = loss + args.replay_loss_weight * replay_auxiliary
                replay_loss_value = float(replay_auxiliary.item())
            (loss / args.meta_batch_size).backward()
            with torch.no_grad():
                query_acc = classifier_accuracy(query_out["logits"], qy, episode)
            batch_stats.append(
                (
                    float(loss.item()),
                    float(classification_loss.item()),
                    float(fssae["total"].item()),
                    query_acc,
                    replay_loss_value,
                )
            )

            if args.replay:
                with torch.no_grad():
                    real_x = torch.cat([sx, qx], dim=0)
                    real_y = torch.cat([sy, qy], dim=0)
                    replay.add(model.srn.encode(real_x), real_y)
        optimizer.step()

        if meta_iteration == 1 or meta_iteration % args.log_every == 0:
            values = np.asarray(batch_stats)
            event = {
                "meta_iteration": meta_iteration,
                "query_ACC": float(values[:, 3].mean()),
                "loss": float(values[:, 0].mean()),
                "classification_loss": float(values[:, 1].mean()),
                "fssae_loss": float(values[:, 2].mean()),
                "replay_auxiliary_loss": float(values[:, 4].mean()),
                "meta_batch_size": args.meta_batch_size,
                "replay": {"enabled": args.replay, **replay.audit()},
            }
            events.append(event)
            print(json.dumps(event), flush=True)

        if meta_iteration % args.validation_every == 0 or meta_iteration == args.meta_iterations:
            validation = select_validation_candidate(
                model,
                args.validation_online_lrs,
                feature_path=args.features,
                feature_payload=feature_payload,
                protocol=protocol,
                adaptation_epochs=args.validation_adaptation_epochs,
                batch_size=args.online_batch_size,
                eval_batch_size=args.eval_batch_size,
                workers=args.workers,
                seed=args.seed + 4000,
                device=device,
                classifier_reset_init=args.classifier_reset_init,
                prediction_class_scope=args.validation_prediction_class_scope,
            )
            record = {"meta_iteration": meta_iteration, **validation}
            validation_events.append(record)
            print(json.dumps({"validation": record}), flush=True)
            step_path = output / f"{run_name}_step{meta_iteration}.pt"
            save_checkpoint(step_path, model, optimizer, config, args, meta_iteration, best_validation_acc)
            if validation["validation_ACC"] > best_validation_acc:
                best_validation_acc = validation["validation_ACC"]
                save_checkpoint(output / f"{run_name}_best.pt", model, optimizer, config, args, meta_iteration, best_validation_acc)
                selection = {
                    "phase": "validation",
                    "selected_meta_iteration": meta_iteration,
                    "selected_checkpoint": str(output / f"{run_name}_best.pt"),
                    "selected_online_lr": validation["selected_online_lr"],
                    "best": validation["best"],
                }
                (output / f"{run_name}_validation_best.json").write_text(
                    json.dumps(selection, indent=2, sort_keys=True)
                )

        metrics_payload = {
            "run": run_name,
            "encoder": feature_payload["encoder"],
            "paper_valid_embeddings": feature_payload["paper_valid"],
            "events": events,
            "validation_events": validation_events,
            "best_validation_ACC": best_validation_acc,
            "arguments": vars(args),
        }
        (output / f"{run_name}_metrics.json").write_text(json.dumps(metrics_payload, indent=2, sort_keys=True))

    save_checkpoint(output / f"{run_name}.pt", model, optimizer, config, args, args.meta_iterations, best_validation_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="clinc150_features/clinc150_minilm_l6_v2.pt")
    parser.add_argument("--protocol", default="clinc150_protocols/domain_holdout_6_4_seed9.json")
    parser.add_argument("--output-dir", default="clinc150_heldout_results")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--meta-iterations", type=int, default=30000)
    parser.add_argument("--ways", type=int, default=5)
    parser.add_argument("--support-shots", type=int, default=2)
    parser.add_argument("--query-shots", type=int, default=5)
    parser.add_argument("--meta-lr", type=float, default=1e-4)
    parser.add_argument("--inner-lr", type=float, default=0.01)
    parser.add_argument("--update-step", type=int, default=5)
    parser.add_argument("--meta-batch-size", type=int, default=1)
    parser.add_argument("--classifier-reset-init", choices=["zero", "kaiming"], default="zero")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--classifier-hidden", type=int, default=512)
    parser.add_argument("--sparsity-target", type=float, default=0.05)
    parser.add_argument("--sparsity-weight", type=float, default=0.005)
    parser.add_argument("--reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--fisher-weight", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=2000, help="Retained for run-manifest compatibility")
    parser.add_argument("--validation-every", type=int, default=2000)
    parser.add_argument("--validation-online-lrs", type=float, nargs="+", default=[0.001, 0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--validation-adaptation-epochs", type=int, default=1)
    parser.add_argument("--validation-prediction-class-scope", choices=["task", "seen"], default="task")
    parser.add_argument("--online-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--replay", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--replay-warmup", type=int, default=100)
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--replay-gap", type=int, default=32)
    parser.add_argument("--replay-rate", type=float, default=0.5)
    parser.add_argument("--replay-loss-weight", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
