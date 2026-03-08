from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Set


@dataclass
class SetCoverInstance:
    n: int
    m: int
    costs: List[float]
    sets: List[Set[int]]

    @classmethod
    def from_file(cls, path: str) -> "SetCoverInstance":
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline()
            if not header:
                raise ValueError("Empty file")

            parts = header.strip().split()
            if len(parts) != 2:
                raise ValueError(f"Bad header in {path}: {header!r}")

            n = int(parts[0])
            m = int(parts[1])

            costs: List[float] = []
            sets: List[Set[int]] = []

            for i in range(m):
                line = f.readline()
                if not line:
                    raise ValueError(f"Unexpected EOF in {path}, set line {i}")

                tokens = line.strip().split()
                if not tokens:
                    raise ValueError(f"Empty set line {i} in {path}")

                c = float(tokens[0])
                elems = set()

                for t in tokens[1:]:
                    e = int(t)
                    if e < 0 or e >= n:
                        raise ValueError(f"Element out of range in {path}: {e}")
                    elems.add(e)

                costs.append(c)
                sets.append(elems)

        return cls(n=n, m=m, costs=costs, sets=sets)


def parse_solution(solution_str: str) -> List[int]:
    solution_str = solution_str.strip()
    if not solution_str:
        return []
    return [int(x) for x in solution_str.split()]


def check_solution(inst: SetCoverInstance, chosen_set_ids: List[int], objective_value: float, eps: float = 1e-8):
    seen = set()
    for sid in chosen_set_ids:
        if sid < 0 or sid >= inst.m:
            return False, f"set id out of range: {sid}"
        if sid in seen:
            return False, f"duplicate set id: {sid}"
        seen.add(sid)

    covered = [False] * inst.n
    total_cost = 0.0

    for sid in chosen_set_ids:
        total_cost += inst.costs[sid]
        for e in inst.sets[sid]:
            covered[e] = True

    uncovered = [i for i, ok in enumerate(covered) if not ok]
    if uncovered:
        return False, f"infeasible: uncovered elements = {len(uncovered)}"

    if not math.isfinite(objective_value):
        return False, "objective is not finite"

    tol = eps * max(1.0, abs(objective_value), abs(total_cost))
    if abs(total_cost - objective_value) > tol:
        return False, f"objective mismatch: csv={objective_value}, real={total_cost}"

    return True, f"OK, cost={total_cost}, sets={len(chosen_set_ids)}"


def main():

    data_folder =  "data"
    results_csv = "results.csv"

    if not os.path.isdir(data_folder):
        print(f"ERROR: folder does not exist: {data_folder}")
        sys.exit(1)

    if not os.path.isfile(results_csv):
        print(f"ERROR: csv does not exist: {results_csv}")
        sys.exit(1)

    ok_count = 0
    fail_count = 0
    skip_count = 0
    total_count = 0

    with open(results_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required_columns = {"test_name", "objective_value", "best_found_time_sec", "solution"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            print(f"ERROR: missing columns in csv: {sorted(missing)}")
            sys.exit(1)

        for row in reader:
            total_count += 1
            test_name = row["test_name"].strip()
            obj_str = row["objective_value"].strip()
            time_str = row["best_found_time_sec"].strip()
            solution_str = row["solution"]

            path = os.path.join(data_folder, test_name)

            if not os.path.isfile(path):
                fail_count += 1
                print(f"✗ {test_name}: instance file not found")
                continue

            if obj_str == "ERROR":
                skip_count += 1
                print(f"• {test_name}: solver wrote ERROR")
                continue

            try:
                objective_value = float(obj_str)
            except Exception:
                fail_count += 1
                print(f"✗ {test_name}: bad objective_value = {obj_str!r}")
                continue

            try:
                best_found_time = float(time_str)
                if best_found_time < 0:
                    fail_count += 1
                    print(f"✗ {test_name}: negative best_found_time_sec = {best_found_time}")
                    continue
            except Exception:
                fail_count += 1
                print(f"✗ {test_name}: bad best_found_time_sec = {time_str!r}")
                continue

            try:
                inst = SetCoverInstance.from_file(path)
                chosen_set_ids = parse_solution(solution_str)
                ok, msg = check_solution(inst, chosen_set_ids, objective_value)

                if ok:
                    ok_count += 1
                    print(f"✓ {test_name}: {msg}, best_found_time_sec={best_found_time}")
                else:
                    fail_count += 1
                    print(f"✗ {test_name}: {msg}, best_found_time_sec={best_found_time}")
            except Exception as e:
                fail_count += 1
                print(f"✗ {test_name}: exception during check: {e}")

    print()
    print(f"Total:  {total_count}")
    print(f"Passed: {ok_count}")
    print(f"Failed: {fail_count}")
    print(f"Skipped:{skip_count}")


if __name__ == "__main__":
    main()
