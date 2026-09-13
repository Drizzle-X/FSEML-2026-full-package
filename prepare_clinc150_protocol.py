import argparse
import json
import random
from pathlib import Path


def make_tasks(domains, selected, seed):
    rng = random.Random(seed)
    ordered_domains = list(selected)
    rng.shuffle(ordered_domains)
    tasks = []
    for domain in ordered_domains:
        intents = list(domains[domain])
        rng.shuffle(intents)
        tasks.extend(intents[start : start + 5] for start in range(0, len(intents), 5))
    return tasks


def build_protocol(domains_path, seed):
    domains = json.loads(Path(domains_path).read_text(encoding="utf-8"))
    test_domains = ["banking", "credit_cards", "travel", "home"]
    train_domains = [name for name in domains if name not in set(test_domains)]
    if len(domains) != 10 or len(train_domains) != 6:
        raise ValueError("Expected the official ten-domain CLINC150 taxonomy")
    if any(len(intents) != 15 for intents in domains.values()):
        raise ValueError("Expected exactly 15 intents in every CLINC150 domain")
    intents = sorted(intent for values in domains.values() for intent in values)
    return {
        "meta_train_domains": train_domains,
        "meta_validation_domains": [],
        "meta_test_domains": test_domains,
        "validation_tasks": make_tasks(domains, train_domains, seed + 1000),
        "test_tasks": make_tasks(domains, test_domains, seed + 2000),
        "intent_to_id": {intent: index for index, intent in enumerate(intents)},
        "intent_to_domain": {
            intent: domain for domain, values in domains.items() for intent in values
        },
        "seed": seed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default="vendor/oos-eval/data/domains.json")
    parser.add_argument("--output", default="clinc150_protocols/domain_holdout_6_4_seed9.json")
    parser.add_argument("--seed", type=int, default=9)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_protocol(args.domains, args.seed), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
