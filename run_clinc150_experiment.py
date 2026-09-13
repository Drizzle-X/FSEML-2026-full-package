from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets.clinc150_continual import CLINCProtocol, EmbeddingDataset
from model.vector_fseml import VectorFSEML, VectorFSSAEConfig


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def masked_loss(logits, labels, allowed):
    allowed = torch.tensor(sorted(allowed), device=logits.device)
    mapping = torch.full((logits.size(1),), -1, dtype=torch.long, device=logits.device)
    mapping[allowed] = torch.arange(len(allowed), device=logits.device)
    return F.cross_entropy(logits.index_select(1, allowed), mapping[labels])


@torch.no_grad()
def evaluate(model, loader, allowed, device):
    model.eval(); allowed = torch.tensor(sorted(allowed), device=device); correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = allowed[model(x)["logits"].index_select(1, allowed).argmax(1)]
        correct += pred.eq(y).sum().item(); total += len(y)
    return correct / max(1, total)


def metrics(matrix, k):
    row = matrix[k - 1, :k]
    acc = float(np.nanmean(row)); la = float(np.nanmean(np.diag(matrix[:k, :k])))
    fm = 0.0 if k == 1 else float(np.mean([np.nanmax(matrix[j:k - 1, j]) - matrix[k - 1, j] for j in range(k - 1)]))
    return {"ACC": acc, "FM": fm, "LA": la}


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    protocol = CLINCProtocol.load(args.protocol)
    features = torch.load(args.features, map_location="cpu", weights_only=False)
    config = VectorFSSAEConfig(input_dim=features["embeddings"].shape[1])
    model = VectorFSEML(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr)
    train_intents = json.loads(Path(args.domains).read_text())
    train_intents = [intent for domain in protocol.meta_train_domains for intent in train_intents[domain]]
    train_data = EmbeddingDataset(args.features, "train", train_intents)
    by_label = {label: torch.where(train_data.y == label)[0].tolist() for label in train_data.y.unique().tolist()}
    labels = sorted(by_label)
    rng = np.random.RandomState(args.seed)
    model.train()
    for step in range(args.meta_steps):
        episode = rng.choice(labels, min(5, len(labels)), replace=False).tolist()
        support_idx, query_idx = [], []
        for label in episode:
            chosen = rng.choice(by_label[label], min(8, len(by_label[label])), replace=False)
            support_idx.extend(chosen[:2]); query_idx.extend(chosen[2:])
        sx, sy = train_data.x[support_idx].to(device), train_data.y[support_idx].to(device)
        qx, qy = train_data.x[query_idx].to(device), train_data.y[query_idx].to(device)
        fast = model.cpn_parameters()
        support = model(sx, fast)
        inner = masked_loss(support["logits"], sy, episode)
        grads = torch.autograd.grad(inner, fast, create_graph=True)
        fast = [p - args.inner_lr * g for p, g in zip(fast, grads)]
        out = model(qx, fast)
        loss = masked_loss(out["logits"], qy, episode) + model.srn.loss(qx, out["latent"], out["reconstruction"], qy)["total"]
        optimizer.zero_grad(); loss.backward(); optimizer.step()


    per_class_capacity = max(1, args.buffer_size // 60)
    buffer = defaultdict(lambda: deque(maxlen=per_class_capacity))
    matrix = np.full((len(protocol.tasks), len(protocol.tasks)), np.nan)
    seen = set(); events = []
    for task_index, intents in enumerate(protocol.tasks):
        current = {protocol.intent_to_id[name] for name in intents}; seen.update(current)
        data = EmbeddingDataset(args.features, "train", intents)
        loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
        online_optimizer = torch.optim.Adam(model.parameters(), lr=args.online_lr)
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            real_x, real_y = x, y
            stored = [item for values in buffer.values() for item in values]
            if args.replay and stored:
                count = min(max(1, int(len(x) * args.replay_rate)), len(stored))
                chosen = random.sample(stored, count)
                latent = torch.stack([item[0] for item in chosen]).to(device)
                replay_x = model.srn.decode(latent).detach()
                replay_y = torch.tensor([item[1] for item in chosen], device=device)
                x, y = torch.cat([x, replay_x]), torch.cat([y, replay_y])
            out = model(x)
            loss = masked_loss(out["logits"], y, seen) + model.srn.loss(x, out["latent"], out["reconstruction"], y)["total"]
            online_optimizer.zero_grad(); loss.backward(); online_optimizer.step()
            with torch.no_grad():
                latent = model.srn.encode(real_x).cpu()
                for z, label in zip(latent, real_y.cpu().tolist()): buffer[int(label)].append((z, int(label)))
        for eval_index in range(task_index + 1):
            test = EmbeddingDataset(args.features, "test", protocol.tasks[eval_index])
            matrix[task_index, eval_index] = evaluate(model, DataLoader(test, batch_size=256), seen, device)
        event = {"task": task_index, **metrics(matrix, task_index + 1)}; events.append(event); print(json.dumps(event))
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    name = f"clinc150_{'fseml_er' if args.replay else 'fseml'}_seed{args.seed}"
    with (output / f"{name}_R_matrix.csv").open("w", newline="") as handle: csv.writer(handle).writerows(matrix)
    payload = {"run": name, "paper_valid_embeddings": features.get("paper_valid", False), "events": events, "final": events[-1], "arguments": vars(args)}
    (output / f"{name}_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    torch.save(model, output / f"{name}.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="clinc150_features/clinc150_embeddings.pt")
    p.add_argument("--protocol", default="clinc150_protocols/domain_holdout_seed9.json")
    p.add_argument("--domains", default="vendor/oos-eval/data/domains.json")
    p.add_argument("--output-dir", default="clinc150_results")
    p.add_argument("--seed", type=int, default=9); p.add_argument("--meta-steps", type=int, default=20000)
    p.add_argument("--meta-lr", type=float, default=5e-4); p.add_argument("--inner-lr", type=float, default=0.05)
    p.add_argument("--online-lr", type=float, default=2e-4); p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--replay", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--buffer-size", type=int, default=1000); p.add_argument("--replay-rate", type=float, default=0.05)
    p.add_argument("--cpu", action="store_true")
    main(p.parse_args())
