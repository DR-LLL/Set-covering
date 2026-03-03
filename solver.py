from __future__ import annotations

import os
import time
import random
import csv
from dataclasses import dataclass
from typing import List, Set, Optional


RANDOM_SEED = 1337
RNG = random.Random(RANDOM_SEED)

#обертка теста 
@dataclass
class SetCoverInstance:
    n: int
    m: int
    costs: List[float]
    sets: List[Set[int]]
    elem_to_sets: List[List[int]] 
    set_ids: List[int]           
    @classmethod
    def from_file(cls, path: str) -> "SetCoverInstance":
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline()
            if not header:
                raise ValueError("Empty file (missing header 'n m').")
            parts = header.strip().split()
            if len(parts) != 2:
                raise ValueError(f"Header must be 'n m', got: {header!r}")

            n = int(parts[0])
            m = int(parts[1])
            if n <= 0 or m <= 0:
                raise ValueError(f"n and m must be positive, got n={n}, m={m}")

            costs: List[float] = []
            sets: List[Set[int]] = []

            for i in range(m):
                line = f.readline()
                if line is None or line == "":
                    raise ValueError(f"Unexpected EOF: expected {m} set lines, got only {i}.")
                tokens = line.strip().split()
                if len(tokens) == 0:
                    raise ValueError(f"Bad set line {i}: empty line.")

                c = float(tokens[0])
                elems: Set[int] = set()
                for t in tokens[1:]:
                    e = int(t)
                    if e < 0 or e >= n:
                        raise ValueError(f"Element index out of range in set {i}: {e} (n={n})")
                    elems.add(e)

                costs.append(c)
                sets.append(elems)

            elem_to_sets: List[List[int]] = [[] for _ in range(n)]
            for si, elems in enumerate(sets):
                for e in elems:
                    elem_to_sets[e].append(si)

            set_ids = list(range(m))
            return cls(n=n, m=m, costs=costs, sets=sets, elem_to_sets=elem_to_sets, set_ids=set_ids)

#обертка где храним решение
@dataclass
class SetCoverSolution:
    chosen_set_ids: List[int]   
    total_cost: float
    is_feasible: bool
    uncovered_count: int


#сама логика солвера
class SetCoverSolver:
    def __init__(self, instance: SetCoverInstance, time_limit_seconds: float):
        if time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be > 0")
        self.instance = instance
        self.time_limit_seconds = float(time_limit_seconds)

    #жадный алгоритм который сортирует по стоимость/мощность
    def greedy_solve(self, instance: Optional[SetCoverInstance] = None) -> SetCoverSolution:
        inst = instance or self.instance
        n, m = inst.n, inst.m
        order = []
        uncovered = [True] * n
        uncovered_count = n

        chosen_local: List[int] = []
        chosen_flag = [False] * m
        total_cost = 0.0

        while uncovered_count > 0:
            best_i = -1
            best_score = -1.0
            for i in range(m):
                if chosen_flag[i]:
                    continue

                gain = 0
                for e in inst.sets[i]:
                    if uncovered[e]:
                        gain += 1
                if gain == 0:
                    continue

                c = inst.costs[i]
                if c <= 0:
                    best_i = i
                    best_score = float("inf")
                    break

                score = gain / c
                if score > best_score:
                    best_score = score
                    best_i = i

            if best_i == -1:
                return SetCoverSolution(
                    chosen_set_ids=[inst.set_ids[i] for i in chosen_local],
                    total_cost=total_cost,
                    is_feasible=False,
                    uncovered_count=uncovered_count,
                )

            chosen_flag[best_i] = True
            chosen_local.append(best_i)
            total_cost += inst.costs[best_i]

            for e in inst.sets[best_i]:
                if uncovered[e]:
                    uncovered[e] = False
                    uncovered_count -= 1

        return SetCoverSolution(
            chosen_set_ids=[inst.set_ids[i] for i in chosen_local],
            total_cost=total_cost,
            is_feasible=True,
            uncovered_count=0,
        )

    def solve(self, instance: Optional[SetCoverInstance] = None) -> SetCoverSolution:
        #стартовое решение - просто жадник
        inst = instance or self.instance
        t0 = time.perf_counter()

        best = self.greedy_solve(inst)

        if (time.perf_counter() - t0) >= self.time_limit_seconds:
            return best

        #будем пытаться улучшать - но в базовой версии пока просто жадник
        while (time.perf_counter() - t0) < self.time_limit_seconds:
            pass

        return best


def run_all_tests_in_folder(folder: str, time_limit_seconds: float = 5.0, results_csv: str = "results.csv"):
    if not os.path.isdir(folder):
        raise ValueError(f"Folder does not exist: {folder}")

    files = sorted(
        fn for fn in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, fn))
    )

    if not files:
        print(f"No files found in folder: {folder}")
        return

    # Открываем CSV один раз и пишем построчно результаты
    with open(results_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        # можно без заголовка, но обычно удобно:
        writer.writerow(["test_name", "objective_value"])

        print(f"Folder: {folder}")
        print(f"Files: {len(files)}")
        print(f"Per-test time limit: {time_limit_seconds:.3f}s")
        print(f"Random seed: {RANDOM_SEED}")
        print(f"CSV output: {results_csv}\n")

        for fn in files:
            path = os.path.join(folder, fn)
            print(f"=== {fn} ===")
            try:
                inst = SetCoverInstance.from_file(path)
                solver = SetCoverSolver(inst, time_limit_seconds=time_limit_seconds)

                t_run = time.perf_counter()
                sol = solver.solve()
                elapsed = time.perf_counter() - t_run

                print(f"n={inst.n}, m={inst.m}")
                print(f"feasible={sol.is_feasible}, uncovered={sol.uncovered_count}")
                print(f"chosen_sets={len(sol.chosen_set_ids)}, total_cost={sol.total_cost}")
                print(f"time_sec={elapsed:.6f}")

                # в CSV пишем имя теста и значение целевой функции
                writer.writerow([fn, sol.total_cost])

            except Exception as e:
                print(f"ERROR: {e}")
                # чтобы в csv было видно, что тест упал:
                writer.writerow([fn, "ERROR"])

            print()


if __name__ == "__main__":
    run_all_tests_in_folder(folder="data", time_limit_seconds=1.0, results_csv="results.csv")
