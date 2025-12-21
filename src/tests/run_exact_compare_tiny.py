# src/run_exact_compare_tiny.py
from pathlib import Path
from time import perf_counter

from tsp.instance import TSPInstance
from tsp.exact import held_karp, brute_force_bnb

BASE_DIR = Path(__file__).resolve().parent.parent
INST_DIR = BASE_DIR / "data" / "instances"

def main():
    for n in [10, 12, 14]:
        for seed in [1, 2, 3]:
            path = INST_DIR / f"tiny_n{n}_seed{seed}.json"
            inst = TSPInstance.load(str(path))

            # Held-Karp timing
            t0 = perf_counter()
            hk_tour, hk_cost = held_karp(inst, start=0)
            t1 = perf_counter()
            hk_sec = t1 - t0

            # Brute force timing (time limit optional)
            if n == 10:
                limit = None
            elif n == 12:
                limit = 30.0
            else:
                limit = 30.0

            t2 = perf_counter()
            bf_tour, bf_cost, timed_out = brute_force_bnb(inst, start=0, time_limit_sec=limit)
            t3 = perf_counter()
            bf_sec = t3 - t2

            bf_status = "TIMEOUT" if timed_out else "DONE"
            bf_match = (abs(bf_cost - hk_cost) < 1e-9) if (bf_tour is not None and not timed_out) else False

            print(f"{path.name}")
            print(f"  Held-Karp OPT: {hk_cost:.6f}   time={hk_sec:.6f}s")
            print(f"  BruteForce({bf_status}): best={bf_cost:.6f}  match_OPT={bf_match}  time={bf_sec:.6f}s")
            print()

if __name__ == "__main__":
    main()
