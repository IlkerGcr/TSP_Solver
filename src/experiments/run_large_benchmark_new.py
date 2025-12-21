# src/experiments/run_large_benchmark_parallel_v2.py
from __future__ import annotations
from pathlib import Path
import sys
import csv
import json
import concurrent.futures
from time import perf_counter
from statistics import mean

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None, desc=None): 
        return iterable

# ---------------------------------------------------------------------------
# Path & Import Setup
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
BASE_DIR = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsp.instance import TSPInstance
from solvers.sa import SAConfig, sa_solve
from solvers.aco import ACOConfig, aco_solve

# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------
INST_DIR = BASE_DIR / "data" / "instances"
OUT_DIR = BASE_DIR / "data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = BASE_DIR / "data" / "best_params.json"

RUNS_CSV = OUT_DIR / "large_runs.csv"
HIST_JSONL = OUT_DIR / "large_histories.jsonl"

DEFAULT_PARAMS = {
    "tiny":   {"sa": {"alpha": 0.95, "steps": 50_000},     "aco": {"beta": 2.0, "rho": 0.5, "ants": 10}},
    "small":  {"sa": {"alpha": 0.99, "steps": 100_000},    "aco": {"beta": 2.0, "rho": 0.5, "ants": 20}},
    "medium": {"sa": {"alpha": 0.995, "steps": 300_000},   "aco": {"beta": 3.0, "rho": 0.1, "ants": 50}},
    "large":  {"sa": {"alpha": 0.9995, "steps": 1_500_000},"aco": {"beta": 3.0, "rho": 0.1, "ants": 30}},
}


def load_active_params():
    """best_params.json okur, bulamazsa default döner."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                params = json.load(f)
            if not params:
                return DEFAULT_PARAMS
            return params
        except Exception as e:
            print(f"Warning: Could not read best_params.json ({e}). Using defaults.")
    return DEFAULT_PARAMS


ACTIVE_PARAMS = load_active_params()


def get_config(n: int, alg_type: str, seed: int):
    """N değerine göre uygun Config nesnesini oluşturur."""
    if n <= 15:
        key = "tiny"
    elif n <= 25:
        key = "small"
    elif n <= 80:
        key = "medium"
    else:
        key = "large"

    params = ACTIVE_PARAMS.get(key, DEFAULT_PARAMS[key])

    if alg_type == "SA":
        p = params["sa"]
        steps = int(p["steps"])
        return SAConfig(
            init_accept_prob=0.8,
            uphill_samples=100,
            cooling_alpha=float(p["alpha"]),
            iters_per_temp=n * 5,
            min_temp=1e-12,
            max_steps=steps,
            log_interval=max(100, steps // 50),
            seed=seed,
            use_step_budget_only=True,
        )

    if alg_type == "ACO":
        p = params["aco"]
        iterations = 300 if n > 50 else 200
        return ACOConfig(
            num_ants=int(p["ants"]),
            alpha=1.0,
            beta=float(p.get("beta", 3.0)),
            rho=float(p.get("rho", 0.1)),
            q=1.0,
            iterations=iterations,
            time_limit_sec=None,
            use_local_search=True,
            local_search_passes=1,
            tau_min=None,
            tau_max=None,
            eps_pheromone=1e-12,
            best_only_deposit=True,
            log_interval=max(1, iterations // 50),
            seed=seed,
        )

    raise ValueError(f"Unknown algo: {alg_type}")


def run_single_task(task_data):
    """Tek bir algoritma koşusunu gerçekleştirir."""
    alg_name, label, n, inst_seed, alg_seed, file_path = task_data

    inst = TSPInstance.load(str(file_path))
    cfg = get_config(n, alg_name, alg_seed)

    t0 = perf_counter()
    if alg_name == "SA":
        res_raw = sa_solve(inst, cfg)
    else:
        res_raw = aco_solve(inst, cfg)
    t1 = perf_counter()

    return {
        "alg_name": alg_name,
        "label": label,
        "n": n,
        "inst_seed": inst_seed,
        "alg_seed": alg_seed,
        "best_cost": float(res_raw["best_cost"]),
        "runtime_sec": t1 - t0,
        "history": res_raw.get("history", []),
    }


def main():
    print(f"Using parameters from: {CONFIG_FILE if CONFIG_FILE.exists() else 'DEFAULT_PARAMS'}")

    configs = [
        ("small", 20, [1, 2, 3]),
        ("medium", 50, [1, 2, 3]),
        ("large", 200, [1, 2, 3]),
    ]
    alg_seeds = [101, 202, 303, 404, 505]

    tasks = []
    print("Preparing tasks...")
    for label, n, inst_seeds in configs:
        for inst_seed in inst_seeds:
            fname = INST_DIR / f"{label}_n{n}_seed{inst_seed}.json"
            if not fname.exists():
                continue
            for alg_seed in alg_seeds:
                tasks.append(("SA", label, n, inst_seed, alg_seed, fname))
                tasks.append(("ACO", label, n, inst_seed, alg_seed, fname))

    print(f"Total tasks: {len(tasks)}")

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_single_task, t) for t in tasks]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Benchmarking"):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"Task failed: {e}")

    results.sort(key=lambda x: (x["n"], x["label"], x["inst_seed"], x["alg_name"]))

    if HIST_JSONL.exists():
        HIST_JSONL.unlink()

    summary_map = {}
    runs_rows = []

    with open(HIST_JSONL, "a", encoding="utf-8") as hist_f:
        for r in results:
            runs_rows.append([
                r["alg_name"], r["label"], r["n"], r["inst_seed"], r["alg_seed"],
                r["best_cost"], r["runtime_sec"],
            ])

            hist_f.write(json.dumps({
                "algorithm": r["alg_name"],
                "label": r["label"],
                "n": r["n"],
                "instance_seed": r["inst_seed"],
                "alg_seed": r["alg_seed"],
                "history": r["history"],
            }) + "\n")

            key = (r["label"], r["n"], r["inst_seed"])
            if key not in summary_map:
                summary_map[key] = {"sa_c": [], "sa_t": [], "aco_c": [], "aco_t": []}

            if r["alg_name"] == "SA":
                summary_map[key]["sa_c"].append(r["best_cost"])
                summary_map[key]["sa_t"].append(r["runtime_sec"])
            else:
                summary_map[key]["aco_c"].append(r["best_cost"])
                summary_map[key]["aco_t"].append(r["runtime_sec"])

    with open(RUNS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "label", "n", "instance_seed", "alg_seed", "best_cost", "runtime_sec"])
        w.writerows(runs_rows)

    SUMMARY_CSV = OUT_DIR / "large_summary.csv"
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "n", "instance_seed", "sa_best", "sa_avg", "sa_time_avg", "aco_best", "aco_avg", "aco_time_avg"])
        for (label, n, s), d in summary_map.items():
            sa_c, aco_c = d["sa_c"], d["aco_c"]
            w.writerow([
                label, n, s,
                min(sa_c) if sa_c else 0, mean(sa_c) if sa_c else 0, mean(d["sa_t"]) if d["sa_t"] else 0,
                min(aco_c) if aco_c else 0, mean(aco_c) if aco_c else 0, mean(d["aco_t"]) if d["aco_t"] else 0,
            ])

    print("\nBenchmark Complete! 🚀")
    print(f"Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
