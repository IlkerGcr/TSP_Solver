# src/experiments/run_grid_search_v3.py
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

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsp.instance import TSPInstance
from solvers.sa import SAConfig, sa_solve
from solvers.aco import ACOConfig, aco_solve

INST_DIR = BASE_DIR / "data" / "instances"
CONFIG_FILE = BASE_DIR / "data" / "best_params.json"

# --- GRID ---
SA_GRIDS = {
    "small":  {"alpha": [0.90, 0.95, 0.99], "steps": [50_000, 100_000]},
    "medium": {"alpha": [0.99, 0.995, 0.999], "steps": [100_000, 300_000]},
    "large":  {"alpha": [0.999, 0.9995, 0.9999], "steps": [500_000, 1_500_000]},
}

ACO_GRIDS = {
    "small":  {"beta": [2.0, 3.0], "rho": [0.1, 0.5], "ants": [10, 20, 30]},
    "medium": {"beta": [2.0, 3.0], "rho": [0.1, 0.5], "ants": [20, 50]},
    "large":  {"beta": [2.0, 3.0, 5.0], "rho": [0.1], "ants": [12, 30, 50]},
}


def evaluate_sa(file_path: Path, alpha: float, steps: int, seed: int):
    inst = TSPInstance.load(str(file_path))
    cfg = SAConfig(
        init_accept_prob=0.8,
        uphill_samples=100,
        cooling_alpha=alpha,
        iters_per_temp=inst.n * 5,
        min_temp=1e-12,
        max_steps=steps,
        log_interval=max(100, steps // 50),
        seed=seed,
        use_step_budget_only=True,
    )
    t0 = perf_counter()
    res = sa_solve(inst, cfg)
    t1 = perf_counter()
    return ("SA", alpha, steps, float(res["best_cost"]), float(t1 - t0))


def evaluate_aco(file_path: Path, beta: float, rho: float, ants: int, seed: int):
    inst = TSPInstance.load(str(file_path))
    cfg = ACOConfig(
        num_ants=ants,
        alpha=1.0,
        beta=beta,
        rho=rho,
        q=1.0,
        iterations=300 if inst.n > 50 else 200,
        time_limit_sec=None,
        use_local_search=True,
        local_search_passes=1,
        tau_min=None,
        tau_max=None,
        eps_pheromone=1e-12,
        best_only_deposit=True,
        log_interval=max(1, (300 if inst.n > 50 else 200) // 50),
        seed=seed,
    )
    t0 = perf_counter()
    res = aco_solve(inst, cfg)
    t1 = perf_counter()
    return ("ACO", beta, rho, ants, float(res["best_cost"]), float(t1 - t0))


def analyze_results(results, alg_type: str, param_names, tol_ratio: float = 0.005):
    grouped = {}
    for r in results:
        if r[0] != alg_type:
            continue
        params = tuple(r[1:1 + len(param_names)])
        cost = float(r[-2])
        runtime = float(r[-1])
        if params not in grouped:
            grouped[params] = {"costs": [], "times": []}
        grouped[params]["costs"].append(cost)
        grouped[params]["times"].append(runtime)

    if not grouped:
        raise RuntimeError(f"No results collected for {alg_type}")

    stats = []
    for params, v in grouped.items():
        avg_cost = mean(v["costs"])
        avg_time = mean(v["times"])
        stats.append((params, avg_cost, avg_time))

    best_cost = min(s[1] for s in stats)
    candidates = [s for s in stats if s[1] <= best_cost * (1.0 + tol_ratio)]
    chosen = min(candidates, key=lambda x: x[2])

    return dict(zip(param_names, chosen[0]))


def run_tuning_for_group(group_name: str, n: int, inst_seeds: list[int]):
    print(f"\n>>> Tuning for {group_name.upper()} (N={n}) <<<")

    tasks = []

    grid_sa = SA_GRIDS[group_name]
    for alpha, steps in itertools.product(grid_sa["alpha"], grid_sa["steps"]):
        for inst_seed in inst_seeds:
            file_path = INST_DIR / f"{group_name}_n{n}_seed{inst_seed}.json"
            if not file_path.exists():
                continue
            for seed in [42, 43]:
                tasks.append((evaluate_sa, file_path, alpha, steps, seed))

    grid_aco = ACO_GRIDS[group_name]
    alg_seeds = [42, 43] if group_name != "large" else [42, 43]
    for beta, rho, ants in itertools.product(grid_aco["beta"], grid_aco["rho"], grid_aco["ants"]):
        for inst_seed in inst_seeds:
            file_path = INST_DIR / f"{group_name}_n{n}_seed{inst_seed}.json"
            if not file_path.exists():
                continue
            for seed in alg_seeds:
                tasks.append((evaluate_aco, file_path, beta, rho, ants, seed))

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
    results = {}

    results["tiny"] = {
        "sa": {"alpha": 0.95, "steps": 50_000},
        "aco": {"beta": 2.0, "rho": 0.5, "ants": 10},
    }

    results["small"] = run_tuning_for_group("small", 20, [1, 2, 3])
    results["medium"] = run_tuning_for_group("medium", 50, [1, 2, 3])
    results["large"] = run_tuning_for_group("large", 200, [1, 2])

    print(f"\nSaving configuration to: {CONFIG_FILE}")
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print("Done. Parameters are now auto-generated!")


if __name__ == "__main__":
    main()
