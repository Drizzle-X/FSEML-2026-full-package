import argparse
import json
import os

from datasets.cifar100_continual import build_cifar100_protocol


def main(args):
    seeds = args.seeds if args.seeds is not None else [args.seed]
    for seed in seeds:
        for classes_per_task in args.classes_per_task:
            protocol = build_cifar100_protocol(
                classes_per_task=classes_per_task,
                seed=seed,
                meta_train_fraction=args.meta_train_fraction,
            )
            path = os.path.join(
                args.output_dir,
                f"cifar100_{classes_per_task}classes_seed{seed}.json",
            )
            protocol.save(path)
            print(json.dumps({
                "path": os.path.abspath(path),
                "seed": seed,
                "classes_per_task": classes_per_task,
                "num_tasks": protocol.num_tasks,
                "meta_train_tasks": len(protocol.meta_train_tasks),
                "meta_test_tasks": len(protocol.meta_test_tasks),
            }, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create reproducible CIFAR100-5/10/20 task manifests.")
    parser.add_argument("--classes-per-task", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Generate several independent protocols; overrides --seed.",
    )
    parser.add_argument("--meta-train-fraction", type=float, default=0.60)
    parser.add_argument("--output-dir", default="cifar100_protocols")
    main(parser.parse_args())
