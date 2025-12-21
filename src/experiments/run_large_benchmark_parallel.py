
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
    def tqdm(iterable, total=None):
        return iterable

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
BASE_DIR = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsp.instance import TSPInstance
from solvers.sa import SAConfig, sa_solve
from solvers.aco import ACOConfig, aco_solve

INST_DIR = BASE_DIR / "data" / "instances"
OUT_DIR = BASE_DIR / "data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS_CSV = OUT_DIR / "large_runs.csv"
HIST_JSONL = OUT_DIR / "large_histories.jsonl"


def run_single_task(task_data):
    """
    Worker function that runs inside a separate process.
    Receives all necessary data to run one algorithm instance.
    """
    alg_name, label, n, inst_seed, alg_seed, file_path = task_data
    
    inst = TSPInstance.load(str(file_path))
    
    result = {}
    
    t0 = perf_counter()
    
    if alg_name == "SA":

        if n <= 20: max_steps = 80_000
        elif n <= 50: max_steps = 200_000
        else: max_steps = 600_000
        
        cfg = SAConfig(
            init_accept_prob=0.8, uphill_samples=100, cooling_alpha=0.995,
            iters_per_temp=500, min_temp=1e-3, max_steps=max_steps,
            log_interval=max_steps // 50, seed=alg_seed
        )
        res_raw = sa_solve(inst, cfg)
        
    else: 
        
        if n <= 20: 
            num_ants = n; iterations = 200
        elif n <= 50: 
            num_ants = 20; iterations = 200
        else: 
            num_ants = 40; iterations = 400 

        cfg = ACOConfig(
            num_ants=num_ants, alpha=1.0, beta=3.0, rho=0.1, q=1.0,
            iterations=iterations, time_limit_sec=None,
            use_local_search=True, local_search_passes=1,
            log_interval=max(1, iterations // 50), seed=alg_seed
        )
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
        "history": res_raw.get("history", [])
    }


def main():

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
                print(f"Skipping missing file: {fname}")
                continue
            
            # Create tasks for both SA and ACO
            for alg_seed in alg_seeds:
                tasks.append(("SA", label, n, inst_seed, alg_seed, fname))
                tasks.append(("ACO", label, n, inst_seed, alg_seed, fname))

    print(f"Total tasks to run: {len(tasks)}")
    
    
    results = []
    
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all tasks
        futures = [executor.submit(run_single_task, t) for t in tasks]
        
        # Monitor progress
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Benchmarking"):
            try:
                res = f.result()
                results.append(res)
            except Exception as e:
                print(f"Task failed: {e}")

    
    print("Saving results...")
    
    # Sort results for cleaner CSV (by N, then Label, then Seed)
    results.sort(key=lambda x: (x["n"], x["label"], x["inst_seed"], x["alg_name"]))
    
    # Prepare Summary Data
    # Key: (label, n, inst_seed) -> {sa_costs:[], aco_costs:[], ...}
    summary_map = {} 
    
    runs_rows = []
    
    if HIST_JSONL.exists():
        HIST_JSONL.unlink()

    with open(HIST_JSONL, "a", encoding="utf-8") as hist_f:
        for r in results:
            # Runs CSV data
            runs_rows.append([
                r["alg_name"], r["label"], r["n"], r["inst_seed"], r["alg_seed"],
                r["best_cost"], r["runtime_sec"]
            ])
            
            # JSONL History data
            hist_entry = {
                "algorithm": r["alg_name"],
                "label": r["label"],
                "n": r["n"],
                "instance_seed": r["inst_seed"],
                "alg_seed": r["alg_seed"],
                "history": r["history"]
            }
            hist_f.write(json.dumps(hist_entry) + "\n")
            
            # Aggregate for Summary
            key = (r["label"], r["n"], r["inst_seed"])
            if key not in summary_map:
                summary_map[key] = {"sa_c":[], "sa_t":[], "aco_c":[], "aco_t":[]}
            
            if r["alg_name"] == "SA":
                summary_map[key]["sa_c"].append(r["best_cost"])
                summary_map[key]["sa_t"].append(r["runtime_sec"])
            else:
                summary_map[key]["aco_c"].append(r["best_cost"])
                summary_map[key]["aco_t"].append(r["runtime_sec"])

    # Write Runs CSV
    with open(RUNS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "label", "n", "instance_seed", "alg_seed", "best_cost", "runtime_sec"])
        w.writerows(runs_rows)
        
    # Write Summary CSV
    SUMMARY_CSV = OUT_DIR / "large_summary_multi_v2.csv"
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "label", "n", "instance_seed",
            "sa_best", "sa_avg", "sa_time_avg",
            "aco_best", "aco_avg", "aco_time_avg"
        ])
        
        for (label, n, s), d in summary_map.items():
            sa_c, sa_t = d["sa_c"], d["sa_t"]
            aco_c, aco_t = d["aco_c"], d["aco_t"]
            
            w.writerow([
                label, n, s,
                min(sa_c) if sa_c else 0, mean(sa_c) if sa_c else 0, mean(sa_t) if sa_t else 0,
                min(aco_c) if aco_c else 0, mean(aco_c) if aco_c else 0, mean(aco_t) if aco_t else 0,
            ])

    print("\nDone! 🚀")
    print(f"Runs: {RUNS_CSV}")
    print(f"Summary: {SUMMARY_CSV}")

if __name__ == "__main__":
    main()