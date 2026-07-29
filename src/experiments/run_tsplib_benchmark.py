"""
Validate SA/ACO against TSPLIB instances with externally published optimal
tour lengths (data/tsplib/known_optima.json), instead of only comparing
against this project's own randomly generated instances.
"""
from __future__ import annotations
from pathlib import Path
import sys
import csv
import json
from time import perf_counter
from statistics import mean

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
BASE_DIR = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsp.tsplib_loader import load_tsplib_instance
from solvers.sa import sa_solve
from solvers.aco import aco_solve
from experiments.run_large_benchmark import get_config

TSPLIB_DIR = BASE_DIR / "data" / "tsplib"
INST_DIR = BASE_DIR / "data" / "instances"
OUT_DIR = BASE_DIR / "data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INSTANCES = ["berlin52", "st70"]
ALG_SEEDS = [101, 202, 303, 404, 505]


def convert_tsplib_instances():
    """Parse each TSPLIB .tsp file once and cache it as a TSPInstance JSON."""
    converted = {}
    for name in INSTANCES:
        tsp_path = TSPLIB_DIR / f"{name}.tsp"
        json_path = INST_DIR / f"tsplib_{name}.json"
        inst = load_tsplib_instance(str(tsp_path))
        inst.save(str(json_path))
        converted[name] = (inst, json_path)
        print(f"Converted {tsp_path.name} -> {json_path.relative_to(BASE_DIR)} (n={inst.n})")
    return converted


def run_one(name, inst, known_optimum):
    rows = []
    sa_costs, aco_costs = [], []

    for seed in ALG_SEEDS:
        sa_cfg = get_config(inst.n, "SA", seed)
        t0 = perf_counter()
        sa_res = sa_solve(inst, sa_cfg)
        sa_time = perf_counter() - t0
        sa_costs.append(sa_res["best_cost"])
        rows.append(["SA", name, inst.n, seed, sa_res["best_cost"], sa_time])

        aco_cfg = get_config(inst.n, "ACO", seed)
        t0 = perf_counter()
        aco_res = aco_solve(inst, aco_cfg)
        aco_time = perf_counter() - t0
        aco_costs.append(aco_res["best_cost"])
        rows.append(["ACO", name, inst.n, seed, aco_res["best_cost"], aco_time])

    sa_best, aco_best = min(sa_costs), min(aco_costs)
    print(f"\n{name} (n={inst.n}, published optimum={known_optimum}):")
    print(f"  SA  best={sa_best:.2f}  gap={100 * (sa_best / known_optimum - 1):.2f}%  "
          f"avg={mean(sa_costs):.2f}")
    print(f"  ACO best={aco_best:.2f}  gap={100 * (aco_best / known_optimum - 1):.2f}%  "
          f"avg={mean(aco_costs):.2f}")

    return rows


def main():
    with open(TSPLIB_DIR / "known_optima.json", "r", encoding="utf-8") as f:
        known_optima = json.load(f)

    converted = convert_tsplib_instances()

    all_rows = []
    for name in INSTANCES:
        inst, _ = converted[name]
        all_rows.extend(run_one(name, inst, known_optima[name]))

    out_path = OUT_DIR / "tsplib_results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "instance", "n", "alg_seed", "best_cost", "runtime_sec"])
        w.writerows(all_rows)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
