# src/experiments/run_tiny_benchmark.py
"""
Run SA and ACO on tiny instances (10/12/14) and compare to exact OPT (Held-Karp).

- OPT is loaded from ground_truth.csv if present.
- If ground_truth.csv is missing for some instance, OPT will run with SA and ACO using HK.

"""

from __future__ import annotations
from pathlib import Path
import sys
import csv
from time import perf_counter
from statistics import mean

# for imports when running
THIS_DIR = Path(__file__).resolve().parent           
SRC_DIR = THIS_DIR.parent                            
BASE_DIR = SRC_DIR.parent                            

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsp.instance import TSPInstance
from tsp.exact import held_karp
from solvers.sa import SAConfig, sa_solve
from solvers.aco import ACOConfig, aco_solve


INST_DIR = BASE_DIR / "data" / "instances"
OUT_DIR = BASE_DIR / "data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUND_TRUTH_CSV = OUT_DIR / "ground_truth.csv"
RUNS_CSV = OUT_DIR / "tiny_runs.csv"
SUMMARY_CSV = OUT_DIR / "tiny_summary.csv"


def load_ground_truth() -> dict[tuple[int, int], tuple[float, float]]:
    """
    Load ground truth OPT values from CSV.

    Returns:
      dict[(n, instance_seed)] = (opt_cost, opt_time_sec)
    """
    gt: dict[tuple[int, int], tuple[float, float]] = {}
    if not GROUND_TRUTH_CSV.exists():
        return gt

    with open(GROUND_TRUTH_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            n = int(row["n"])
            s = int(row["instance_seed"])
            opt = float(row["opt_cost"])
            t = float(row["opt_time_sec"])
            gt[(n, s)] = (opt, t)
    return gt


def run_sa(inst: TSPInstance, alg_seed: int) -> tuple[float, float]:
    """
    Run Simulated Annealing once.
    Returns: (best_cost, runtime_sec)
    """
    cfg = SAConfig(
        init_accept_prob=0.8,
        uphill_samples=100,
        cooling_alpha=0.995,
        iters_per_temp=500,
        min_temp=1e-3,
        max_steps=50_000,
        log_interval=1000,
        seed=alg_seed,
    )

    t0 = perf_counter()
    res = sa_solve(inst, cfg)
    t1 = perf_counter()
    return float(res["best_cost"]), (t1 - t0)


def run_aco(inst: TSPInstance, alg_seed: int) -> tuple[float, float]:
    """
    Run Ant Colony Optimization once.
    Returns: (best_cost, runtime_sec)
    """
    n = inst.n
    cfg = ACOConfig(
        num_ants=n,              # standard choice for small instances
        alpha=1.0,
        beta=3.0,
        rho=0.2,
        q=1.0,
        iterations=200,
        time_limit_sec=None,

        # ACO + 2-opt hybrid (better for tiny)
        use_local_search=True,
        local_search_passes=1,

        # pheromone clamp (optional)
        tau_min=None,
        tau_max=None,

        log_interval=20,
        seed=alg_seed,
    )

    t0 = perf_counter()
    res = aco_solve(inst, cfg)
    t1 = perf_counter()
    return float(res["best_cost"]), (t1 - t0)


def main() -> None:
    # Instances to evaluate
    tiny_ns = [10, 12, 14]
    inst_seeds = [1, 2, 3]

    # Multiple stochastic runs per instance
    alg_seeds = [101, 202, 303, 404, 505]

    gt = load_ground_truth()

    runs_rows: list[list] = []
    summary_rows: list[list] = []

    for n in tiny_ns:
        for s in inst_seeds:
            fname = INST_DIR / f"tiny_n{n}_seed{s}.json"
            if not fname.exists():
                raise FileNotFoundError(f"Missing instance file: {fname}")

            inst = TSPInstance.load(str(fname))

            # Get OPT (first CSV if not calculate )
            key = (n, s)
            if key in gt:
                opt, opt_time = gt[key]
            else:
                t0 = perf_counter()
                _, opt = held_karp(inst, start=0)
                t1 = perf_counter()
                opt_time = t1 - t0

            # SA runs 
            sa_costs: list[float] = []
            sa_times: list[float] = []
            for alg_seed in alg_seeds:
                best_cost, rt = run_sa(inst, alg_seed)
                ratio = best_cost / opt
                runs_rows.append([
                    "SA", n, s, alg_seed,
                    opt, opt_time,
                    best_cost, rt, ratio
                ])
                sa_costs.append(best_cost)
                sa_times.append(rt)

            # ACO runs
            aco_costs: list[float] = []
            aco_times: list[float] = []
            for alg_seed in alg_seeds:
                best_cost, rt = run_aco(inst, alg_seed)
                ratio = best_cost / opt
                runs_rows.append([
                    "ACO", n, s, alg_seed,
                    opt, opt_time,
                    best_cost, rt, ratio
                ])
                aco_costs.append(best_cost)
                aco_times.append(rt)

            # Her bir instance için özet
            summary_rows.append([
                n, s,
                opt, opt_time,

                min(sa_costs), mean(sa_costs), mean(sa_times),
                min(aco_costs), mean(aco_costs), mean(aco_times),

                (min(sa_costs) / opt), (mean(sa_costs) / opt),
                (min(aco_costs) / opt), (mean(aco_costs) / opt),
            ])

            print(f"[OK] tiny_n{n}_seed{s}: OPT={opt:.6f}")

    # Write per run CSV
    with open(RUNS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "algorithm", "n", "instance_seed", "alg_seed",
            "opt_cost", "opt_time_sec",
            "best_cost", "runtime_sec", "ratio"
        ])
        w.writerows(runs_rows)

    # Write per instance summary CSV
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "n", "instance_seed",
            "opt_cost", "opt_time_sec",
            "sa_best_cost", "sa_avg_cost", "sa_avg_time_sec",
            "aco_best_cost", "aco_avg_cost", "aco_avg_time_sec",
            "sa_best_ratio", "sa_avg_ratio",
            "aco_best_ratio", "aco_avg_ratio",
        ])
        w.writerows(summary_rows)

    print("\nSaved:")
    print(" -", RUNS_CSV)
    print(" -", SUMMARY_CSV)
    print(" (Ground truth used from:", GROUND_TRUTH_CSV if GROUND_TRUTH_CSV.exists() else "Held-Karp (computed)", ")")

if __name__ == "__main__":
    main()
