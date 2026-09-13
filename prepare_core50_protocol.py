from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

from datasets.core50_continual import NBATCH


def digest(values):
    return hashlib.sha256(",".join(map(str, values)).encode()).hexdigest()


def main(args):
    root = Path(args.metadata_root)
    lookup = pickle.load((root / "LUP.pkl").open("rb"))
    labels = pickle.load((root / "labels.pkl").open("rb"))
    batches = []
    for index in range(NBATCH[args.scenario]):
        y = [int(v) for v in labels[args.scenario][args.run][index]]
        ids = lookup[args.scenario][args.run][index]
        batches.append({"batch": index, "samples": len(ids), "classes": sorted(set(y)), "index_sha256": digest(ids)})
    payload = {"dataset": "CORe50", "scenario": args.scenario, "run": args.run, "num_batches": len(batches), "batches": batches}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True)); print(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--metadata-root", default="vendor/core50_official")
    p.add_argument("--scenario", choices=sorted(NBATCH), default="nicv2_391"); p.add_argument("--run", type=int, default=0)
    p.add_argument("--output", default="core50_protocols/nicv2_391_run0.json"); main(p.parse_args())

