"""
Command-line entry point for the TSP solver.

Installed as the `tsp` console script (see pyproject.toml). Usage:

    tsp solve --instance data/instances/tsplib_berlin52.json --algo aco
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from tsp.instance import TSPInstance
from tsp.exact import held_karp
from solvers.sa import SAConfig, sa_solve
from solvers.aco import ACOConfig, aco_solve

EXACT_SIZE_WARNING_THRESHOLD = 16  # Held-Karp is O(n^2 * 2^n); this gets slow fast.


def solve_exact(instance: TSPInstance, seed: int):
    if instance.n > EXACT_SIZE_WARNING_THRESHOLD:
        print(
            f"Warning: exact solver is O(n^2 * 2^n); n={instance.n} may take a long time.",
            file=sys.stderr,
        )
    tour, cost = held_karp(instance, start=0)
    return {"best_tour": tour, "best_cost": cost}


def solve_sa(instance: TSPInstance, seed: int):
    cfg = SAConfig(
        init_accept_prob=0.8,
        uphill_samples=100,
        cooling_alpha=0.995,
        iters_per_temp=instance.n * 5,
        min_temp=1e-12,
        max_steps=300_000,
        log_interval=1000,
        seed=seed,
        use_step_budget_only=True,
    )
    return sa_solve(instance, cfg)


def solve_aco(instance: TSPInstance, seed: int):
    cfg = ACOConfig(
        num_ants=max(20, instance.n // 2),
        alpha=1.0,
        beta=3.0,
        rho=0.1,
        q=1.0,
        iterations=300,
        use_local_search=True,
        local_search_passes=1,
        log_interval=10,
        seed=seed,
    )
    return aco_solve(instance, cfg)


SOLVERS = {"exact": solve_exact, "sa": solve_sa, "aco": solve_aco}


def cmd_solve(args: argparse.Namespace) -> int:
    instance = TSPInstance.load(args.instance)
    solve_fn = SOLVERS[args.algo]

    t0 = perf_counter()
    result = solve_fn(instance, args.seed)
    runtime = perf_counter() - t0

    print(f"algorithm : {args.algo}")
    print(f"n         : {instance.n}")
    print(f"best_cost : {result['best_cost']:.4f}")
    print(f"runtime   : {runtime:.4f}s")

    if args.output:
        payload = {
            "algorithm": args.algo,
            "n": instance.n,
            "seed": args.seed,
            "best_cost": result["best_cost"],
            "best_tour": result["best_tour"],
            "runtime_sec": runtime,
        }
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved result to: {args.output}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsp", description="TSP solver CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    solve_parser = subparsers.add_parser("solve", help="Solve a single TSP instance")
    solve_parser.add_argument("--instance", required=True, help="Path to a TSPInstance JSON file")
    solve_parser.add_argument("--algo", choices=sorted(SOLVERS), required=True)
    solve_parser.add_argument("--seed", type=int, default=42)
    solve_parser.add_argument("--output", default=None, help="Optional path to write the result as JSON")
    solve_parser.set_defaults(func=cmd_solve)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
