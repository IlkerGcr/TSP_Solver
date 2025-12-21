#Finds best params and SAVES them to best_params.json 

from __future__ import annotations
from pathlib import Path
import sys
import itertools
import concurrent.futures
import json
from time import perf_counter
from statistics import mean

# Path setup
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
BASE_DIR = SRC_DIR.parent
sys.path.append(str(SRC_DIR))

from tsp.instance import TSPInstance
from solvers.sa import SAConfig, sa_solve
from solvers.aco import ACOConfig, aco_solve

INST_DIR = BASE_DIR / "data" / "instances"
CONFIG_FILE = BASE_DIR / "data" / "best_params.json"

# --- DEFINING THE GRID ---
SA_GRIDS = {
    "small":  {"alpha": [0.90, 0.95, 0.99], "steps": [50_000, 100_000]},
    "medium": {"alpha": [0.99, 0.995, 0.999], "steps": [100_000, 300_000]},
    "large":  {"alpha": [0.999, 0.9995, 0.9999], "steps": [500_000, 1_500_000]}
}

ACO_GRIDS = {
    "small":  {"beta": [2.0, 3.0], "rho": [0.1, 0.5], "ants": [10, 20, 30]},
    "medium": {"beta": [2.0, 3.0], "rho": [0.1, 0.5], "ants": [20, 50]},
    "large":  {"beta": [2.0, 3.0, 5.0], "rho": [0.1], "ants": [12, 30, 50]}
}

def evaluate_sa(file_path, alpha, steps, seed):
    inst = TSPInstance.load(str(file_path))
    cfg = SAConfig(
        init_accept_prob=0.8, uphill_samples=100, cooling_alpha=alpha,
        iters_per_temp=inst.n * 5, min_temp=1e-4, max_steps=steps, seed=seed
    )
    t0 = perf_counter()
    res = sa_solve(inst, cfg)
    return ("SA", alpha, steps, res["best_cost"], perf_counter() - t0)

def evaluate_aco(file_path, beta, rho, ants, seed):
    inst = TSPInstance.load(str(file_path))
    cfg = ACOConfig(
        num_ants=ants, alpha=1.0, beta=beta, rho=rho, q=1.0,
        iterations=300 if inst.n > 50 else 200, 
        use_local_search=True, local_search_passes=1, seed=seed
    )
    t0 = perf_counter()
    res = aco_solve(inst, cfg)
    return ("ACO", beta, rho, ants, res["best_cost"], perf_counter() - t0)

def analyze_results(results, alg_type, param_names):
    grouped = {}
    for r in results:
        if r[0] != alg_type: continue
        params = tuple(r[1 : 1+len(param_names)]) 
        cost = r[-2]
        if params not in grouped: grouped[params] = []
        grouped[params].append(cost)

    best_params = None
    min_avg = float('inf')
    for params, costs in grouped.items():
        avg = mean(costs)
        if avg < min_avg:
            min_avg = avg
            best_params = params
            
    return dict(zip(param_names, best_params))

def run_tuning_for_group(group_name, n, inst_seed_to_use):
    print(f"\n>>> Tuning for {group_name.upper()} (N={n}) <<<")
    file_path = INST_DIR / f"{group_name}_n{n}_seed{inst_seed_to_use}.json"
    
    tasks = []
    
    # SA Tasks
    grid_sa = SA_GRIDS[group_name]
    for alpha, steps in itertools.product(grid_sa["alpha"], grid_sa["steps"]):
        for seed in [42, 43]: tasks.append((evaluate_sa, file_path, alpha, steps, seed))
            
    # ACO Tasks
    grid_aco = ACO_GRIDS[group_name]
    seeds = [42] if group_name == "large" else [42, 43]
    for beta, rho, ants in itertools.product(grid_aco["beta"], grid_aco["rho"], grid_aco["ants"]):
        for seed in seeds: tasks.append((evaluate_aco, file_path, beta, rho, ants, seed))

    print(f"   Running {len(tasks)} tasks...")
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(t[0], *t[1:]) for t in tasks]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    best_sa = analyze_results(results, "SA", ["alpha", "steps"])
    best_aco = analyze_results(results, "ACO", ["beta", "rho", "ants"])
    
    print(f"   🏆 Best SA: {best_sa}")
    print(f"   🏆 Best ACO: {best_aco}")
    
    return {"sa": best_sa, "aco": best_aco}

def main():
    # 1. Tüm gruplar için tuning yap
    # Tiny (N=10-14) için tuning yapmıyoruz, Small'a yakın varsayıyoruz ama manual ekleyeceğiz.
    
    results = {}
    
    # Tiny için varsayılan (Tune etmiyoruz, çok hızlı zaten)
    results["tiny"] = {
        "sa": {"alpha": 0.95, "steps": 50000},
        "aco": {"beta": 2.0, "rho": 0.5, "ants": 10}
    }
    
    results["small"] = run_tuning_for_group("small", 20, 1)
    results["medium"] = run_tuning_for_group("medium", 50, 1)
    results["large"] = run_tuning_for_group("large", 200, 1)

    # 2. JSON Olarak Kaydet
    print(f"\nSaving configuration to: {CONFIG_FILE}")
    with open(CONFIG_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print("Done. Parameters are now auto-generated!")

if __name__ == "__main__":
    main()