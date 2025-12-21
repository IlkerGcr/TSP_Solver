# src/run_aco_smoke.py
from tsp.instance import TSPInstance
from solvers.aco import ACOConfig, aco_solve

if __name__ == "__main__":
    # Same square instance
    inst = TSPInstance.from_coords([(0,0), (0,1), (1,1), (1,0)])

    cfg = ACOConfig(
        num_ants=4,          # one ant per city, just for demo
        alpha=1.0,
        beta=3.0,
        rho=0.3,
        q=1.0,
        iterations=50,
        time_limit_sec=None,
        use_local_search=False,   # baseline ACO first
        local_search_passes=1,
        log_interval=5,
        seed=123,
    )

    result = aco_solve(inst, cfg)
    print("Best cost:", result["best_cost"])
    print("Best tour:", result["best_tour"])
    print("Iterations:", result["iterations_done"])
    print("Runtime (s):", result["runtime_sec"])
    print("History (first 5):", result["history"][:5])
    