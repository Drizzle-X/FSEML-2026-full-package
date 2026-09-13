import json
import tempfile
import unittest
from pathlib import Path

from continual_metrics import class_order, metrics_from_matrix
from run_paper_five_runs import commands
from run_paper_table_iii import clinc150_commands, core50_commands
from summarize_paper_five_runs import main as summarize_main
from summarize_paper_table_iii import main as summarize_table_iii_main
from verify_results import compare_actual

ROOT = Path(__file__).resolve().parents[1]

class FiveRunProtocolTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs" / "paper_main_five_runs.json").read_text(encoding="utf-8"))

    def test_seed_set_and_all_cifar_protocols(self):
        self.assertEqual(self.config["seeds"], [9, 19, 29, 39, 49])
        orders = set()
        for seed in self.config["seeds"]:
            seed_order = None
            for size in (5, 10, 20):
                path = ROOT / "cifar100_protocols" / f"cifar100_{size}classes_seed{seed}.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(payload["seed"], seed)
                self.assertEqual(payload["classes_per_task"], size)
                self.assertEqual(sorted(payload["class_order"]), list(range(100)))
                self.assertEqual(payload["num_meta_train_tasks"] + payload["num_meta_test_tasks"], 100 // size)
                seed_order = seed_order or payload["class_order"]
                self.assertEqual(payload["class_order"], seed_order)
            orders.add(tuple(seed_order))
        self.assertEqual(len(orders), 5)

    def test_command_matrix_is_complete(self):
        generated = list(commands(self.config, "python3", ["cifar100", "omniglot"], ["fseml", "fseml-er"],
            self.config["seeds"], "/data/cifar", "/data/omni", ROOT / "paper_runs"))
        self.assertEqual(len(generated), 90)
        self.assertEqual(sum(phase == "train" for phase, _ in generated), 40)
        self.assertEqual(sum(phase == "evaluate" for phase, _ in generated), 40)
        self.assertEqual(sum(phase == "select" for phase, _ in generated), 10)
        omniglot = [(phase, command) for phase, command in generated if "omniglot" in " ".join(command)]
        self.assertTrue(all("evaluate_classification.py" not in command for _, command in omniglot))
        self.assertTrue(all("--selection" in command for phase, command in omniglot if phase == "evaluate"))
        table_iii = []
        for method in ("fseml", "fseml-er"):
            table_iii.extend(core50_commands("python3", "/data/core50", ROOT / "paper_runs/table_iii", method))
            table_iii.extend(clinc150_commands("python3", "/data/clinc.pt", ROOT / "paper_runs/table_iii", method))
        self.assertEqual(len(table_iii), 10)

    def test_table_iii_has_five_protocols_and_fifty_commands(self):
        seeds = self.config["seeds"]
        commands = []
        core_orders = set()
        clinc_orders = set()
        for seed in seeds:
            core_path = ROOT / "core50_heldout_splits" / f"split_seed{seed}.json"
            clinc_path = ROOT / "clinc150_protocols" / f"domain_holdout_6_4_seed{seed}.json"
            core = json.loads(core_path.read_text(encoding="utf-8"))
            clinc = json.loads(clinc_path.read_text(encoding="utf-8"))
            self.assertEqual(core["seed"], seed)
            self.assertEqual(len(core["meta_train_classes"]), 30)
            self.assertEqual(len(core["meta_validation_classes"]), 10)
            self.assertEqual(len(core["meta_test_classes"]), 10)
            self.assertEqual(clinc["seed"], seed)
            self.assertEqual([len(clinc[key]) for key in ("meta_train_domains", "meta_validation_domains", "meta_test_domains")], [6, 0, 4])
            self.assertEqual(len(clinc["validation_tasks"]), 18)
            self.assertEqual(len(clinc["test_tasks"]), 12)
            self.assertTrue(all(len(task) == 5 for task in clinc["validation_tasks"] + clinc["test_tasks"]))
            train_domains = set(clinc["meta_train_domains"])
            test_domains = set(clinc["meta_test_domains"])
            self.assertFalse(train_domains & test_domains)
            validation_domains = [
                {clinc["intent_to_domain"][intent] for intent in task}
                for task in clinc["validation_tasks"]
            ]
            task_domains = [
                {clinc["intent_to_domain"][intent] for intent in task}
                for task in clinc["test_tasks"]
            ]
            self.assertTrue(all(len(domains) == 1 and next(iter(domains)) in train_domains for domains in validation_domains))
            self.assertTrue(all(len(domains) == 1 and next(iter(domains)) in test_domains for domains in task_domains))
            self.assertEqual([next(iter(domains)) for domains in task_domains][0::3], [next(iter(domains)) for domains in task_domains][1::3])
            self.assertEqual([next(iter(domains)) for domains in task_domains][1::3], [next(iter(domains)) for domains in task_domains][2::3])
            core_orders.add(tuple(value for group in core["semantic_groups"].values() for value in group))
            clinc_orders.add(tuple(value for task in clinc["validation_tasks"] + clinc["test_tasks"] for value in task))
            for method in ("fseml", "fseml-er"):
                commands.extend(core50_commands("python3", "/data/core50", ROOT / "paper_runs/table_iii", method, seed))
                commands.extend(clinc150_commands("python3", "/data/clinc.pt", ROOT / "paper_runs/table_iii", method, seed))
        self.assertEqual(len(commands), 50)
        self.assertEqual(len(core_orders), 5)
        self.assertEqual(len(clinc_orders), 5)

    def test_five_run_summary_is_verifier_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"; results.mkdir()
            omniglot_results = root / "omniglot"; omniglot_results.mkdir()
            for method in ("fseml", "fseml-er"):
                for seed in self.config["seeds"]:
                    payload = {"seed": seed, "dataset": "Omniglot-1", "final": {"ACC": 0.75, "FM": 0.03, "LA": 0.8}}
                    (omniglot_results / f"omniglot_{method}_seed{seed}_metrics.json").write_text(json.dumps(payload), encoding="utf-8")
            for method in ("fseml", "fseml-er"):
                for size in (5, 10, 20):
                    for seed in self.config["seeds"]:
                        payload = {"seed": seed, "classes_per_task": size, "final": {"ACC": 0.5, "FM": 0.1, "LA": 0.6}}
                        (results / f"cifar100_{size}_{method}_seed{seed}_metrics.json").write_text(json.dumps(payload), encoding="utf-8")
            output = root / "summary.json"
            self.assertEqual(summarize_main(["--results-dir", str(results), "--omniglot-results-dir", str(omniglot_results), "--output", str(output)]), 0)
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["table-i-fseml"]["ACC"], {"mean": 75.0, "std": 0.0})
            self.assertEqual(summary["table-ii-fseml-c5"]["ACC"], {"mean": 50.0, "std": 0.0})
            paper = json.loads((ROOT / "expected_results.json").read_text(encoding="utf-8"))
            self.assertTrue(compare_actual(paper, summary))

    def test_omniglot_matrix_metrics_and_seeded_order(self):
        matrix = [[0.9, None, None], [0.8, 0.7, None], [0.7, 0.6, 0.8]]
        metrics = metrics_from_matrix(matrix)
        self.assertAlmostEqual(metrics["ACC"], 0.7)
        self.assertAlmostEqual(metrics["FM"], 0.15)
        self.assertAlmostEqual(metrics["LA"], 0.8)
        self.assertEqual(class_order(20, 9), class_order(20, 9))
        self.assertNotEqual(class_order(20, 9), class_order(20, 19))

    def test_table_iii_five_seed_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for method in ("fseml", "fseml-er"):
                target = root / f"core50_{method}"; target.mkdir()
                target = root / f"clinc150_{method}"; target.mkdir()
            for seed in self.config["seeds"]:
                payload = {"final_ACC": 0.5, "forgetting": 0.1, "learning_ACC": 0.6}
                for method in ("fseml", "fseml-er"):
                    (root / f"core50_{method}" / f"test_seed{seed}.json").write_text(json.dumps(payload), encoding="utf-8")
                    name = f"clinc150_fseml_er_decoder_seed{seed}_test.json" if method == "fseml-er" else f"clinc150_fseml_seed{seed}_test.json"
                    (root / f"clinc150_{method}" / name).write_text(json.dumps(payload), encoding="utf-8")
            output = root / "summary.json"
            self.assertEqual(summarize_table_iii_main(["--root", str(root), "--output", str(output)]), 0)
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["table-iii-core50-fseml"]["ACC"], {"mean": 50.0, "std": 0.0})

if __name__ == "__main__": unittest.main()
