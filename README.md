# TSP Solver & Benchmark Suite

A Python framework for solving the Traveling Salesman Problem (TSP) with multiple algorithms, and benchmarking/tuning them against generated problem instances.

## Algorithms implemented
- **Exact** — Held–Karp dynamic programming (optimal, for small instances) and a Branch & Bound brute-force solver with nearest-neighbor bounding and symmetry pruning.
- **Simulated Annealing (SA)** — with automatic initial-temperature estimation and 2-opt moves.
- **Ant Colony Optimization (ACO)** — configurable pheromone/heuristic weighting, evaporation, pheromone clamping, and optional 2-opt local search on the iteration-best tour.

## Structure
```
src/tsp/         core data structures: TSPInstance (coords + distance matrix, JSON load/save),
                 tour cost, O(1)-delta 2-opt moves, exact solvers
src/solvers/     SA and ACO solver implementations
src/experiments/ instance generation, parallel grid-search tuning, benchmark runners
data/instances/  pre-generated problem instances (10 to 200 cities)
data/results/    benchmark run outputs
data/best_params.json   tuned hyperparameters found by grid search
```

## Running it
```bash
# Generate problem instances
python src/experiments/generate_instances.py

# Tune SA/ACO parameters via parallel grid search (uses ProcessPoolExecutor)
python src/experiments/run_grid_search.py
```

## Highlights
- Exact Held–Karp DP solver for verifying optimal solutions on small instances.
- Hyperparameter tuning is parallelized across CPU cores and picks the fastest configuration within a cost tolerance of the best found (not just the lowest-cost one).
- Instances range from 10 to 200 cities across four size categories (tiny/small/medium/large).

## Tech stack
Python 3 (dataclasses, `concurrent.futures`), no external dependencies.
