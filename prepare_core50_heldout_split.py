import argparse
import json
import random
from pathlib import Path


def main(args):
    rng = random.Random(args.seed)
    train, validation, test = [], [], []
    groups = {}
    for semantic_group in range(10):
        labels = list(range(semantic_group * 5, semantic_group * 5 + 5))
        rng.shuffle(labels)
        groups[str(semantic_group)] = labels
        train.extend(labels[:3])
        validation.append(labels[3])
        test.append(labels[4])
    payload = {
        "dataset": "CORe50",
        "protocol": "stratified-held-out-class-adaptation-v1",
        "seed": args.seed,
        "semantic_groups": groups,
        "meta_train_classes": sorted(train),
        "meta_validation_classes": sorted(validation),
        "meta_test_classes": sorted(test),
        "support_sessions": [1, 2, 4, 5],
        "meta_train_query_sessions": [6, 8, 9, 11],
        "official_test_sessions": [3, 7, 10],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--output", default="core50_heldout_splits/split_seed9.json")
    main(parser.parse_args())
