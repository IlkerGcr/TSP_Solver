"""
Solve a TSP instance with tour-history recording enabled and render an
animated GIF of the tour untangling over iterations, synced with the
convergence curve. Used to generate the README's hero image.

Palette follows the project's dataviz reference (see conversation/plan):
ACO = blue, SA = orange, on a fixed light surface (static GIFs embedded in
Markdown can't respond to a viewer's dark/light setting, so one mode is
chosen deliberately rather than left to chance).
"""
from __future__ import annotations
from pathlib import Path
import sys
import argparse

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
BASE_DIR = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsp.instance import TSPInstance
from solvers.sa import SAConfig, sa_solve
from solvers.aco import ACOConfig, aco_solve

PLOTS_DIR = BASE_DIR / "data" / "plots"
TARGET_FRAMES = 50

# --- dataviz reference palette (light surface, fixed for static/GIF embeds) ---
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
            log_interval=max(1, max_steps // TARGET_FRAMES),
            seed=seed,
            use_step_budget_only=True,
            record_tour_history=True,
        )
        return sa_solve(instance, cfg)

    if algo == "aco":
        iterations = 300
        cfg = ACOConfig(
            num_ants=max(20, instance.n // 2),
            alpha=1.0,
            beta=3.0,
            rho=0.1,
            q=1.0,
            iterations=iterations,
            use_local_search=True,
            local_search_passes=1,
            log_interval=max(1, iterations // TARGET_FRAMES),
            seed=seed,
            record_tour_history=True,
        )
        return aco_solve(instance, cfg)

    raise ValueError(f"Unknown algorithm: {algo}")


def trim_after_convergence(tour_history, cost_history, hold: int = 6):
    """
    tour_history and cost_history are recorded at the same log-interval
    cadence (same length, same step values), so they can be trimmed
    together by index. Cut the long flat tail once the best cost stops
    improving, keeping a short hold so the "final reveal" still registers.
    """
    assert len(tour_history) == len(cost_history)
    last_improvement = 0
    for i in range(1, len(cost_history)):
        if cost_history[i][1] < cost_history[i - 1][1] - 1e-9:
            last_improvement = i
    cutoff = min(len(cost_history), last_improvement + hold + 1)
    return tour_history[:cutoff], cost_history[:cutoff]


def make_animation(instance: TSPInstance, result: dict, algo: str, title_prefix: str, out_path: Path):
    tour_history, cost_history = trim_after_convergence(result["tour_history"], result["history"])
    color = SERIES[algo]
    xs_all = [c[0] for c in instance.coords]
    ys_all = [c[1] for c in instance.coords]

    fig, (ax_tour, ax_curve) = plt.subplots(1, 2, figsize=(13, 6), facecolor=SURFACE)
    for ax in (ax_tour, ax_curve):
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_color(GRIDLINE)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)

    ax_tour.set_xlim(min(xs_all) - 60, max(xs_all) + 60)
    ax_tour.set_ylim(min(ys_all) - 60, max(ys_all) + 60)
    ax_tour.set_aspect("equal")
    (tour_line,) = ax_tour.plot([], [], "-", color=color, linewidth=2, zorder=2, solid_capstyle="round")
    ax_tour.scatter(xs_all, ys_all, color=INK_PRIMARY, edgecolors=SURFACE, linewidths=1.2, s=45, zorder=3)
    tour_title = ax_tour.set_title("", fontsize=12, color=INK_PRIMARY)

    steps = [s for s, _ in cost_history]
    costs = [c for _, c in cost_history]
    ax_curve.set_xlim(min(steps), max(steps))
    pad = (max(costs) - min(costs)) * 0.08 or 1.0
    ax_curve.set_ylim(min(costs) - pad, max(costs) + pad)
    ax_curve.set_xlabel("iteration / step", color=INK_MUTED, fontsize=10)
    ax_curve.set_ylabel("best cost so far", color=INK_MUTED, fontsize=10)
    ax_curve.set_title("Convergence", fontsize=12, color=INK_PRIMARY)
    (curve_line,) = ax_curve.plot([], [], "-", color=color, linewidth=2, zorder=2)
    (curve_dot,) = ax_curve.plot([], [], "o", color=color, markersize=9,
                                   markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)

    fig.suptitle(f"{title_prefix} — {LABELS[algo]}", fontsize=14, color=INK_PRIMARY, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    def update(frame_idx):
        # tour_history and cost_history are aligned by index (same log-interval cadence).
        step, tour = tour_history[frame_idx]
        xs = [instance.coords[c][0] for c in tour] + [instance.coords[tour[0]][0]]
        ys = [instance.coords[c][1] for c in tour] + [instance.coords[tour[0]][1]]
        tour_line.set_data(xs, ys)

        curve_line.set_data(steps[: frame_idx + 1], costs[: frame_idx + 1])
        curve_dot.set_data([steps[frame_idx]], [costs[frame_idx]])

        tour_title.set_text(f"cost = {costs[frame_idx]:.2f}  (n={instance.n}, step {step})")
        return tour_line, curve_line, curve_dot, tour_title

    # Hold on the final frame for a beat so the GIF doesn't loop too abruptly.
    hold_frames = 8
    frame_indices = list(range(len(tour_history))) + [len(tour_history) - 1] * hold_frames

    anim = FuncAnimation(fig, lambda i: update(frame_indices[i]), frames=len(frame_indices), blit=False)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=8))
    plt.close(fig)
    print(f"Saved: {out_path}  ({len(tour_history)} tour frames, {out_path.stat().st_size / 1024:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Render an animated GIF of a tour untangling over iterations.")
    parser.add_argument("--instance", required=True)
    parser.add_argument("--algo", choices=["sa", "aco"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    inst_path = Path(args.instance)
    instance = TSPInstance.load(str(inst_path))
    result = solve(instance, args.algo, args.seed)

    label = args.label or inst_path.stem
    out_path = PLOTS_DIR / f"{inst_path.stem}_{args.algo}_animated.gif"
    make_animation(instance, result, args.algo, label, out_path)


if __name__ == "__main__":
    main()
