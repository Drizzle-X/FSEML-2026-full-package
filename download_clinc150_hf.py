from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def import_huggingface_datasets():


    project_root = str(Path(__file__).resolve().parent)
    original_path = list(sys.path)
    sys.path = [entry for entry in sys.path if entry not in ("", project_root)]
    sys.modules.pop("datasets", None)
    try:
        from datasets import load_dataset
    finally:
        sys.path = original_path
    return load_dataset


def decode(feature, value):
    return feature.int2str(value) if hasattr(feature, "int2str") and isinstance(value, int) else str(value)


def main(args):
    load_dataset = import_huggingface_datasets()
    dataset = load_dataset(args.repository, split=args.hf_split)
    features = dataset.features
    required = {"text", "domain", "intent", "split"}
    missing = sorted(required.difference(features))
    if missing:
        raise ValueError(f"Hugging Face CLINC150 mirror is missing columns: {missing}")

    output = {name: [] for name in ("train", "val", "test", "oos_train", "oos_val", "oos_test")}
    domains = defaultdict(set)
    split_aliases = {"validation": "val", "oos_validation": "oos_val"}
    for row in dataset:
        raw_split = decode(features["split"], row["split"])
        split = split_aliases.get(raw_split, raw_split)
        domain = decode(features["domain"], row["domain"])
        intent = decode(features["intent"], row["intent"])
        if ":" in intent:
            intent_domain, intent = intent.split(":", 1)
            if domain in ("None", "oos", "0") and intent_domain != "oos":
                domain = intent_domain
        if split not in output:
            raise ValueError(f"Unexpected CLINC150 split {split!r}")
        is_oos = split.startswith("oos_") or intent == "oos" or domain == "oos"
        label = "oos" if is_oos else intent
        output[split].append([str(row["text"]), label])
        if not is_oos:
            domains[domain].add(intent)

    expected = {"train": 15000, "val": 3000, "test": 4500, "oos_train": 100, "oos_val": 100, "oos_test": 1000}
    counts = {name: len(rows) for name, rows in output.items()}
    if counts != expected:
        raise ValueError(f"Unexpected CLINC150 split counts: {counts}; expected {expected}")
    domain_payload = {name: sorted(values) for name, values in sorted(domains.items())}
    if len(domain_payload) != 10 or any(len(values) != 15 for values in domain_payload.values()):
        raise ValueError(
            "Hugging Face domain taxonomy does not match official CLINC150: "
            f"{ {name: len(values) for name, values in domain_payload.items()} }"
        )

    data_path = Path(args.output_dir) / "data_full.json"
    domains_path = Path(args.output_dir) / "domains.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(output, indent=2))
    domains_path.write_text(json.dumps(domain_payload, indent=2, sort_keys=True))
    print(json.dumps({"source": args.repository, "counts": counts, "domains": len(domain_payload)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default="contemmcm/clinc150")
    parser.add_argument("--hf-split", default="complete")
    parser.add_argument("--output-dir", default="vendor/oos-eval/data")
    main(parser.parse_args())
