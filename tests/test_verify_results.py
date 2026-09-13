import unittest
from pathlib import Path
from verify_results import compare_actual, validate_manuscript

class VerifyResultsTest(unittest.TestCase):
    def setUp(self):
        self.spec = {"schema_version": 2, "manuscript_results": [{"id":"paper", "table":"II", "dataset":"CIFAR100-5", "method":"FSEML", "metrics":{"ACC":{"value":77.21}}}]}

    def test_manuscript_schema(self):
        self.assertEqual(validate_manuscript(self.spec), [])

    def test_matching_result_passes(self):
        self.assertEqual(compare_actual(self.spec, {"paper": {"ACC": 77.215}}), [])

    def test_out_of_tolerance_result_fails(self):
        self.assertIn("expected 77.21", compare_actual(self.spec, {"paper": {"ACC": 80.0}})[0])

    def test_unknown_result_id_fails(self):
        self.assertIn("unknown manuscript result id", compare_actual(self.spec, {"unknown": {"ACC": 77.21}})[0])

    def test_public_result_scope_is_tables_i_to_iii(self):
        import json
        from pathlib import Path
        specification = json.loads(Path(__file__).resolve().parents[1].joinpath("expected_results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(specification["manuscript_results"]), 12)
        self.assertEqual({row["table"] for row in specification["manuscript_results"]}, {"I", "II", "III"})
        self.assertEqual({row["method"] for row in specification["manuscript_results"]}, {"FSEML", "FSEML-ER"})

    def test_training_defaults_match_manuscript(self):
        source = Path(__file__).resolve().parents[1].joinpath("mrcl_classification.py").read_text(encoding="utf-8")
        for fragment in (
            "default=20000)", "default=5e-4)", "default=0.1)",
            '"--replay-gap", type=int, default=960',
            '"--replay-rate", type=float, default=0.05',
            '"--replay-buffer-size", type=int, default=1000',
        ):
            self.assertIn(fragment, source)

if __name__ == "__main__": unittest.main()
