"""Evaluate the extract.py -> planner.py pipeline against data/eval_set.json.

For every case: run the request through extract_constraints(), then build a
schedule from the *extracted* constraints. Reports two independent metrics:

  - extraction accuracy: does every field named in the case's `expected`
    dict exactly match what extract.py returned? (fields not mentioned in
    `expected` are not checked -- the eval set only asserts what a request
    unambiguously implies.) Also reported per field.
  - plan constraint-satisfaction rate: does the schedule built from the
    *extracted* constraints actually honor those same constraints (no ride
    taller than max_height_cm, none with an excluded tag, nothing scheduled
    outside [arrival_time, end_time))? This checks planner.py's own
    correctness, independent of whether extraction was itself accurate.

Usage:
    python src/evaluate.py
"""

import dataclasses
import datetime as dt
import json

from extract import Constraints, ConstraintValidationError, extract_constraints
from planner import (
    DEFAULT_ARRIVAL,
    DEFAULT_END,
    build_schedule,
    load_model,
    load_rides,
    _parse_hhmm,
)

EVAL_SET_PATH = "data/eval_set.json"

# Fixed so the plan constraint-satisfaction check (which depends on month and
# day-of-week via _monthly_defaults) is reproducible run to run, rather than
# drifting with dt.date.today().
EVAL_DATE = dt.date(2022, 7, 15)

FIELDS = ["arrival_time", "end_time", "max_height_cm", "exclude_tags", "prefer_tags"]


def _values_match(actual, expected) -> bool:
    if isinstance(expected, list):
        return isinstance(actual, list) and set(actual) == set(expected)
    return actual == expected


def _extraction_field_results(actual: dict, expected: dict) -> dict[str, bool]:
    return {field: _values_match(actual[field], expected[field]) for field in expected}


def _schedule_violations(schedule: list[dict], constraints: Constraints, ride_lookup: dict) -> list[str]:
    arrival = _parse_hhmm(constraints.arrival_time or DEFAULT_ARRIVAL)
    end = _parse_hhmm(constraints.end_time or DEFAULT_END)
    exclude = set(constraints.exclude_tags)

    violations = []
    for stop in schedule:
        ride = ride_lookup[stop["ride"]]
        stop_minutes = _parse_hhmm(stop["time"])

        if stop_minutes < arrival or stop_minutes >= end:
            violations.append(
                f"{stop['ride']} at {stop['time']} is outside "
                f"[{constraints.arrival_time or DEFAULT_ARRIVAL}, {constraints.end_time or DEFAULT_END})"
            )
        if (
            constraints.max_height_cm is not None
            and ride["min_height_cm"] is not None
            and ride["min_height_cm"] > constraints.max_height_cm
        ):
            violations.append(
                f"{stop['ride']} requires {ride['min_height_cm']}cm, "
                f"exceeding max_height_cm={constraints.max_height_cm}"
            )
        bad_tags = exclude & set(ride["tags"])
        if bad_tags:
            violations.append(f"{stop['ride']} has excluded tag(s) {sorted(bad_tags)}")

    return violations


def evaluate_case(case: dict, rides: list[dict], booster, ride_lookup: dict) -> dict:
    expected = case["expected"]

    try:
        constraints = extract_constraints(case["request"])
    except ConstraintValidationError as e:
        # extract.py's own validation rejected what the model returned (e.g. a
        # malformed time). Record it as a failed case rather than crashing
        # the whole evaluation run.
        return {
            "id": case["id"],
            "request": case["request"],
            "expected": expected,
            "actual": {"error": str(e)},
            "field_results": {},
            "extraction_ok": False,
            "constraint_ok": False,
            "violations": ["extraction raised ConstraintValidationError, no schedule built"],
        }

    actual = dataclasses.asdict(constraints)
    field_results = _extraction_field_results(actual, expected)
    extraction_ok = all(field_results.values())

    schedule = build_schedule(constraints, rides=rides, booster=booster, date=EVAL_DATE)
    violations = _schedule_violations(schedule, constraints, ride_lookup)
    constraint_ok = not violations

    return {
        "id": case["id"],
        "request": case["request"],
        "expected": expected,
        "actual": actual,
        "field_results": field_results,
        "extraction_ok": extraction_ok,
        "constraint_ok": constraint_ok,
        "violations": violations,
    }


def run_evaluation(eval_set_path: str = EVAL_SET_PATH) -> list[dict]:
    with open(eval_set_path, encoding="utf-8") as f:
        eval_set = json.load(f)

    rides = load_rides()
    booster = load_model()
    ride_lookup = {r["name"]: r for r in rides}

    results = []
    cases = eval_set["cases"]
    for i, case in enumerate(cases, 1):
        print(f"Running case {i}/{len(cases)} (id={case['id']})...")
        results.append(evaluate_case(case, rides, booster, ride_lookup))
    return results


def print_report(results: list[dict]) -> None:
    n = len(results)
    extraction_hits = sum(r["extraction_ok"] for r in results)
    constraint_hits = sum(r["constraint_ok"] for r in results)

    print("\n=== Summary ===")
    print(f"Extraction accuracy:          {extraction_hits}/{n} ({100 * extraction_hits / n:.1f}%)")
    print(f"Plan constraint-satisfaction: {constraint_hits}/{n} ({100 * constraint_hits / n:.1f}%)")

    print("\n=== Extraction accuracy by field ===")
    for field in FIELDS:
        relevant = [r["field_results"][field] for r in results if field in r["field_results"]]
        if not relevant:
            print(f"{field:<15} (not exercised by any case)")
            continue
        hits = sum(relevant)
        print(f"{field:<15} {hits}/{len(relevant)} ({100 * hits / len(relevant):.1f}%)")

    failures = [r for r in results if not r["extraction_ok"] or not r["constraint_ok"]]
    print(f"\n=== Failures ({len(failures)}/{n}) ===")
    if not failures:
        print("None.")
        return

    for r in failures:
        print(f"\n--- Case {r['id']} ---")
        print(f"Request:  {r['request']}")
        print(f"Expected: {r['expected']}")
        print(f"Actual:   {r['actual']}")
        if not r["extraction_ok"]:
            bad_fields = [f for f, ok in r["field_results"].items() if not ok]
            print(f"Extraction mismatch on: {bad_fields}")
        if not r["constraint_ok"]:
            for v in r["violations"]:
                print(f"Constraint violation: {v}")


def main() -> None:
    results = run_evaluation()
    print_report(results)


if __name__ == "__main__":
    main()
