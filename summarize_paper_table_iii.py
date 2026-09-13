#!/usr/bin/env python3
"""Aggregate the five-seed Table III outputs for verify_results.py."""
import argparse
import json
import statistics
from pathlib import Path

SEEDS = (9, 19, 29, 39, 49)

def metrics(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"ACC": 100.0 * payload["final_ACC"], "FM": 100.0 * payload["forgetting"], "LA": 100.0 * payload["learning_ACC"]}

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("paper_runs/table_iii"))
    p.add_argument("--output", type=Path, default=Path("paper_runs/table_iii_summary.json"))
    args = p.parse_args(argv)
    patterns = {
        "table-iii-core50-fseml": "core50_fseml/test_seed{seed}.json",
        "table-iii-core50-fseml-er": "core50_fseml-er/test_seed{seed}.json",
        "table-iii-clinc150-fseml": "clinc150_fseml/clinc150_fseml_seed{seed}_test.json",
        "table-iii-clinc150-fseml-er": "clinc150_fseml-er/clinc150_fseml_er_decoder_seed{seed}_test.json",
    }
    result = {}
    for result_id, pattern in patterns.items():
        rows = [metrics(args.root / pattern.format(seed=seed)) for seed in SEEDS]
        result[result_id] = {
            name: {
                "mean": statistics.fmean(row[name] for row in rows),
                "std": statistics.stdev(row[name] for row in rows),
            }
            for name in ("ACC", "FM", "LA")
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output.resolve())
    return 0

if __name__ == "__main__": raise SystemExit(main())
