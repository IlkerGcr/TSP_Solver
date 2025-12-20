"""
Run SA and ACO on larger instances (n=20/50/200) for evaluation.

Outputs:
- large_runs.csv          (per-run: best_cost, runtime)
- large_histories.json   (per-run history arrays as JSON lines)

Note: No OPT / approximation ratio here (ground truth only exists for tiny).
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

from tsp.instance import TSPInstance
from solvers.sa import SAConfig, sa_solve
from solvers.aco import ACOConfig, aco_solve


OUT_DIR = BASE_DIR / "data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INST_DIR = BASE_DIR / "data" / "instances"

RUNS_CSV = OUT_DIR / "large_runs.csv"
HIST_JSONL = OUT_DIR / "large_histories.jsonl"


def run_sa(inst: TSPInstance, alg_seed: int, n: int) -> dict:
    """
    Run SA once and return dict with best_cost, runtime, history.
    We scale max_steps with n to keep some fairness.
    """
    # Heuristic step budget scaling
    # n=20 -> 80k, n=50 -> 200k, n=200 -> 600k (adjust later if too slow)
    if n <= 20:
        max_steps = 80_000
    elif n <= 50:
        max_steps = 200_000
    else:
        max_steps = 600_000

    cfg = SAConfig(           # SA PARAMETRELERI
        init_accept_prob=0.8,
        uphill_samples=100,
        cooling_alpha=0.995,
        iters_per_temp=500,
        min_temp=1e-3,
        max_steps=max_steps,
        log_interval=max_steps // 50,  
        seed=alg_seed,
    )

    t0 = perf_counter()
    res = sa_solve(inst, cfg)
    t1 = perf_counter()

    return {
        "best_cost": float(res["best_cost"]),
        "runtime_sec": (t1 - t0),
        "history": res.get("history", []),
        "config": cfg,
    }


def run_aco(inst: TSPInstance, alg_seed: int, n: int) -> dict:
    """
    Run ACO once and return dict with best_cost, runtime, history.
    We scale ants/iterations a bit with n but keep it bounded.
    """
    # Keep ants reasonable for large n (n=200 -> 50 ants instead of 200)
    if n <= 20:
        num_ants = n
        iterations = 300
    elif n <= 50:
        num_ants = 30
        iterations = 400
    else:
        num_ants = 50
        iterations = 600

    cfg = ACOConfig(         # ACO PARAMETRELERI
        num_ants=num_ants,
        alpha=1.0,
        beta=3.0,
        rho=0.2,
        q=1.0,
        iterations=iterations,
        time_limit_sec=None,

        # For large n, local search can be expensive; started with 1 pass.
        use_local_search=True,
        local_search_passes=1,

        # Optional clamp (Şuanlık kapalı tutuyorum. Lazım olursa diye)
        tau_min=None,
        tau_max=None,

        log_interval=max(1, iterations // 50),  
        seed=alg_seed,
    )

    t0 = perf_counter()
    res = aco_solve(inst, cfg)
    t1 = perf_counter()

    return {
        "best_cost": float(res["best_cost"]),
        "runtime_sec": (t1 - t0),
        "history": res.get("history", []),
        "config": cfg,
    }


def main() -> None:
    # Large instance groups 
    configs = [
        ("small", 20, [1, 2, 3]),
        ("medium", 50, [1, 2, 3]),
        ("large", 200, [1, 2, 3]),
    ]

    # Multiple stochastic runs per instance
    alg_seeds = [101, 202, 303, 404, 505]

    runs_rows: list[list] = []
    summary_rows: dict[tuple[str, int, int], dict] = {}

    # Reset history file
    if HIST_JSONL.exists():
        HIST_JSONL.unlink()

    with open(HIST_JSONL, "a", encoding="utf-8") as hist_f:
        for label, n, inst_seeds in configs:
            for inst_seed in inst_seeds:
                fname = INST_DIR / f"{label}_n{n}_seed{inst_seed}.json"
                if not fname.exists():
                    raise FileNotFoundError(f"Missing instance file: {fname}")

                inst = TSPInstance.load(str(fname))

                # Collect per-instance results to summarize
                sa_costs, sa_times = [], []
                aco_costs, aco_times = [], []

                for alg_seed in alg_seeds:
                    # *** SA ***
                    sa_res = run_sa(inst, alg_seed, n)
                    runs_rows.append([
                        "SA", label, n, inst_seed, alg_seed,
                        sa_res["best_cost"], sa_res["runtime_sec"]
                    ])
                    sa_costs.append(sa_res["best_cost"])
                    sa_times.append(sa_res["runtime_sec"])

                    hist_f.write(json.dumps({
                        "algorithm": "SA",
                        "label": label,
                        "n": n,
                        "instance_seed": inst_seed,
                        "alg_seed": alg_seed,
                        "history": sa_res["history"],
                    }) + "\n")

                    # *** ACO ***
                    aco_res = run_aco(inst, alg_seed, n)
                    runs_rows.append([
                        "ACO", label, n, inst_seed, alg_seed,
                        aco_res["best_cost"], aco_res["runtime_sec"]
                    ])
                    aco_costs.append(aco_res["best_cost"])
                    aco_times.append(aco_res["runtime_sec"])

                    hist_f.write(json.dumps({
                        "algorithm": "ACO",
                        "label": label,
                        "n": n,
                        "instance_seed": inst_seed,
                        "alg_seed": alg_seed,
                        "history": aco_res["history"],
                    }) + "\n")

                # Save summary for this instance
                summary_rows[(label, n, inst_seed)] = {
                    "sa_best_cost": min(sa_costs),
                    "sa_avg_cost": mean(sa_costs),
                    "sa_avg_time": mean(sa_times),
                    "aco_best_cost": min(aco_costs),
                    "aco_avg_cost": mean(aco_costs),
                    "aco_avg_time": mean(aco_times),
                }

                print(f"[OK] {label}_n{n}_seed{inst_seed}")

    # Write per run CSV
    with open(RUNS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "algorithm", "label", "n", "instance_seed", "alg_seed",
            "best_cost", "runtime_sec"
        ])
        w.writerows(runs_rows)

    # Write summary CSV
    summary_path = OUT_DIR / "large_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "label", "n", "instance_seed",
            "sa_best_cost", "sa_avg_cost", "sa_avg_time_sec",
            "aco_best_cost", "aco_avg_cost", "aco_avg_time_sec",
        ])
        for (label, n, inst_seed), d in summary_rows.items():
            w.writerow([
                label, n, inst_seed,
                d["sa_best_cost"], d["sa_avg_cost"], d["sa_avg_time"],
                d["aco_best_cost"], d["aco_avg_cost"], d["aco_avg_time"],
            ])

    print("\nSaved:")
    print(" -", RUNS_CSV)
    print(" -", summary_path)
    print(" -", HIST_JSONL)


if __name__ == "__main__":
    main()
