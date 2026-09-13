#!/usr/bin/env python3
"""Validate result JSON files against expected_results.json (stdlib only)."""
import argparse, json, math, sys
from pathlib import Path

def validate_manuscript(specification):
    errors, seen = [], set()
    records = specification.get("manuscript_results")
    if specification.get("schema_version") != 2 or not isinstance(records, list):
        return ["expected-results file must have schema_version=2 and a manuscript_results list"]
    for record in records:
        record_id = record.get("id", "<missing-id>")
        if record_id in seen: errors.append(f"{record_id}: duplicate manuscript id")
        seen.add(record_id)
        for required in ("table", "dataset", "method", "metrics"):
            if required not in record: errors.append(f"{record_id}: missing {required}")
        if not isinstance(record.get("metrics"), dict) or not record.get("metrics"):
            errors.append(f"{record_id}: metrics must be a non-empty object")
    return errors

def compare_actual(specification, actual, default_atol=0.01):
    """Compare an ID-to-metrics JSON object with configured expected values."""
    errors = []
    expected = {record["id"]: record for record in specification["manuscript_results"]}
    for record_id, metrics in actual.items():
        if record_id not in expected:
            errors.append(f"{record_id}: unknown manuscript result id"); continue
        for name, observed in metrics.items():
            target_spec = expected[record_id]["metrics"].get(name)
            if not isinstance(target_spec, dict) or "value" not in target_spec:
                errors.append(f"{record_id}: {name} is not a scalar manuscript metric"); continue
            target = target_spec["value"]
            tolerance = max(default_atol, target_spec.get("std", 0.0))
            if isinstance(observed, dict) and "mean" in observed:
                observed = observed["mean"]
            if not isinstance(observed, (int, float)) or not math.isfinite(float(observed)):
                errors.append(f"{record_id}: {name} must be a finite number"); continue
            if not math.isclose(observed, target, rel_tol=0.0, abs_tol=tolerance):
                errors.append(f"{record_id}: {name} expected {target} +/- {tolerance}, got {observed}")
    return errors

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, default=Path(__file__).with_name("expected_results.json"))
    parser.add_argument("--actual", type=Path, help="Optional ID-to-metrics JSON from a new reproduction run.")
    parser.add_argument("--atol", type=float, default=0.01, help="Minimum absolute tolerance in manuscript units.")
    args = parser.parse_args(argv)
    try: specification = json.loads(args.expected.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot read {args.expected}: {exc}", file=sys.stderr); return 2
    errors = validate_manuscript(specification)
    if args.actual:
        try: actual = json.loads(args.actual.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"ERROR: cannot read {args.actual}: {exc}", file=sys.stderr); return 2
        if not isinstance(actual, dict): errors.append("actual-results file must be an ID-to-metrics object")
        else: errors.extend(compare_actual(specification, actual, args.atol))
    if errors:
        for error in errors: print(f"FAIL: {error}", file=sys.stderr)
        print(f"Validation failed: {len(errors)} error(s).", file=sys.stderr); return 1
    print(f"Validated {len(specification['manuscript_results'])} expected result record(s)."); return 0

if __name__ == "__main__": raise SystemExit(main())
