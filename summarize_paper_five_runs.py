#!/usr/bin/env python3
"""Aggregate five Omniglot/CIFAR-100 runs into Table I/II mean/std metrics."""
import argparse
import json
import statistics
from pathlib import Path

SEEDS = (9, 19, 29, 39, 49)

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=Path("paper_runs/results/cifar100"))
    p.add_argument("--omniglot-results-dir", type=Path, default=Path("paper_runs/results/omniglot"))
    p.add_argument("--output", type=Path, default=Path("paper_runs/cifar100_five_run_summary.json"))
    args = p.parse_args(argv)
    summary = {}
    for method in ("fseml", "fseml-er"):
        rows = []
        for seed in SEEDS:
            path = args.omniglot_results_dir / f"omniglot_{method}_seed{seed}_metrics.json"
            if not path.is_file(): raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["seed"] != seed or payload["dataset"] != "Omniglot-1":
                raise ValueError(f"metadata mismatch: {path}")
            rows.append({metric: 100.0 * float(payload["final"][metric]) for metric in ("ACC", "FM", "LA")})
        summary[f"table-i-{method}"] = {
            metric: {"mean": statistics.fmean(row[metric] for row in rows), "std": statistics.stdev(row[metric] for row in rows)}
            for metric in ("ACC", "FM", "LA")
        }
    for method in ("fseml", "fseml-er"):
        for size in (5, 10, 20):
            rows = []
            for seed in SEEDS:
                path = args.results_dir / f"cifar100_{size}_{method}_seed{seed}_metrics.json"
                if not path.is_file(): raise FileNotFoundError(path)
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload["seed"] != seed or payload["classes_per_task"] != size:
                    raise ValueError(f"metadata mismatch: {path}")
                rows.append({metric: 100.0 * float(payload["final"][metric]) for metric in ("ACC", "FM", "LA")})
            result_id = f"table-ii-{method}-c{size}"
            summary[result_id] = {
                metric: {"mean": statistics.fmean(row[metric] for row in rows), "std": statistics.stdev(row[metric] for row in rows)}
                for metric in ("ACC", "FM", "LA")
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output.resolve())
    return 0

if __name__ == "__main__": raise SystemExit(main())
