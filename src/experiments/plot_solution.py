"""
Solve a TSP instance and render a two-panel figure: the found tour drawn
over the city map, and the best-cost-vs-iteration convergence curve.
Used to generate the README's static result images (see
make_tour_animation.py for the animated GIF version).

Palette: dataviz reference palette, light surface (ACO = blue, SA = orange).
"""
from __future__ import annotations
from pathlib import Path
import sys
import argparse

import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
BASE_DIR = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsp.instance import TSPInstance
from tsp.tour import tour_length
from solvers.sa import SAConfig, sa_solve
from solvers.aco import ACOConfig, aco_solve

PLOTS_DIR = BASE_DIR / "data" / "plots"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SERIES = {"aco": "#2a78d6", "sa": "#eb6834"}
LABELS = {"aco": "Ant Colony Optimization", "sa": "Simulated Annealing"}


def solve(instance: TSPInstance, algo: str, seed: int):
    if algo == "sa":
        # Scale the cooling rate to the step budget so the schedule always
        # finishes annealing (temperature down to ~0.2% of its start) within it.
        max_steps = 300_000
        iters_per_temp = instance.n * 5
        num_coolings = max(1, max_steps // iters_per_temp)
        cfg = SAConfig(
            init_accept_prob=0.8,
            uphill_samples=100,
            cooling_alpha=0.002 ** (1 / num_coolings),
            iters_per_temp=iters_per_temp,
            min_temp=1e-12,
            max_steps=max_steps,
            log_interval=500,
            seed=seed,
            use_step_budget_only=True,
        )
        return sa_solve(instance, cfg)

    if algo == "aco":
        cfg = ACOConfig(
            num_ants=max(20, instance.n // 2),
            alpha=1.0,
            beta=3.0,
            rho=0.1,
            q=1.0,
            iterations=300,
            use_local_search=True,
            local_search_passes=1,
            log_interval=1,
            seed=seed,
        )
        return aco_solve(instance, cfg)

    raise ValueError(f"Unknown algorithm: {algo}")


def plot_solution(instance: TSPInstance, result: dict, algo: str, title_prefix: str, out_path: Path):
    tour = result["best_tour"]
    cost = result["best_cost"]
    history = result["history"]
    color = SERIES[algo]

    fig, (ax_tour, ax_curve) = plt.subplots(1, 2, figsize=(13, 6), facecolor=SURFACE)
    for ax in (ax_tour, ax_curve):
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_color(GRIDLINE)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)

    xs = [instance.coords[c][0] for c in tour] + [instance.coords[tour[0]][0]]
    ys = [instance.coords[c][1] for c in tour] + [instance.coords[tour[0]][1]]
    ax_tour.plot(xs, ys, "-", color=color, linewidth=2, zorder=2, solid_capstyle="round")
    ax_tour.scatter(
        [c[0] for c in instance.coords], [c[1] for c in instance.coords],
        color=INK_PRIMARY, edgecolors=SURFACE, linewidths=1.2, zorder=3, s=45,
    )
    ax_tour.set_title(f"{title_prefix} — {LABELS[algo]}\ncost = {cost:.2f}  (n={instance.n})",
                       fontsize=12, color=INK_PRIMARY)
    ax_tour.set_xlabel("x", color=INK_MUTED, fontsize=10)
    ax_tour.set_ylabel("y", color=INK_MUTED, fontsize=10)
    ax_tour.axis("equal")

    steps, costs = zip(*history)
    ax_curve.plot(steps, costs, color=color, linewidth=2)
    ax_curve.set_title("Convergence", fontsize=12, color=INK_PRIMARY)
    ax_curve.set_xlabel("iteration / step", color=INK_MUTED, fontsize=10)
    ax_curve.set_ylabel("best cost so far", color=INK_MUTED, fontsize=10)

    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # sanity check: the plotted cost should match an independent recompute
    assert abs(tour_length(tour, instance) - cost) < 1e-6


def main():
    parser = argparse.ArgumentParser(description="Solve an instance and plot the tour + convergence curve.")
    parser.add_argument("--instance", required=True, help="Path to a TSPInstance JSON file")
    parser.add_argument("--algo", choices=["sa", "aco"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", default=None, help="Title prefix (defaults to the instance filename)")
    args = parser.parse_args()

    inst_path = Path(args.instance)
    instance = TSPInstance.load(str(inst_path))
    result = solve(instance, args.algo, args.seed)

    label = args.label or inst_path.stem
    out_path = PLOTS_DIR / f"{inst_path.stem}_{args.algo}.png"
    plot_solution(instance, result, args.algo, label, out_path)


if __name__ == "__main__":
    main()
