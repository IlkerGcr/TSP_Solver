# src/run_sa_smoke.py
from tsp.instance import TSPInstance
from solvers.sa import SAConfig, sa_solve

if __name__ == "__main__":
    # Simple square instance
    inst = TSPInstance.from_coords([(0,0), (0,1), (1,1), (1,0)])

    cfg = SAConfig(
        init_accept_prob=0.8,
        uphill_samples=50,
        cooling_alpha=0.99,
        iters_per_temp=50,
        min_temp=1e-3,
        max_steps=2000,
        log_interval=50,
        seed=42,
    )

    result = sa_solve(inst, cfg)
    print("Best cost:", result["best_cost"])
    print("Best tour:", result["best_tour"])
    print("Steps:", result["steps"])
    print("Runtime (s):", result["runtime_sec"])
    print("History (first 5):", result["history"][:5])
