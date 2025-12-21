from __future__ import annotations
from pathlib import Path
import sys
import csv
from time import perf_counter

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
BASE_DIR = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsp.instance import TSPInstance
from tsp.exact import held_karp

INST_DIR = BASE_DIR / "data" / "instances"
OUT_DIR = BASE_DIR / "data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "ground_truth.csv"


def main() -> None:
    tiny_ns = [10, 12, 14]
    inst_seeds = [1, 2, 3]  

    rows = []
    for n in tiny_ns:
        for s in inst_seeds:
            path = INST_DIR / f"tiny_n{n}_seed{s}.json"
            inst = TSPInstance.load(str(path))

            t0 = perf_counter()
            _, opt = held_karp(inst, start=0)
            t1 = perf_counter()

            rows.append([n, s, float(opt), t1 - t0])

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "instance_seed", "opt_cost", "opt_time_sec"])
        w.writerows(rows)

    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()
