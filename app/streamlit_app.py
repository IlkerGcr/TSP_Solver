"""
Streamlit demo for the TSP solver suite.

Run locally with:
    pip install -e ".[demo]"
    streamlit run app/streamlit_app.py

Solves the selected instance once with tour-history recording enabled, then
hands the full sequence of (step, tour, cost) snapshots to a Plotly figure
with native frame animation (Play/Pause + scrubber) — the tour visibly
untangles in sync with the convergence curve, no manual timing loop needed.
"""
from __future__ import annotations
from pathlib import Path
import sys

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
BASE_DIR = THIS_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsp.instance import TSPInstance
from solvers.sa import SAConfig, sa_solve
from solvers.aco import ACOConfig, aco_solve

INST_DIR = BASE_DIR / "data" / "instances"
TARGET_FRAMES = 50

# --- dataviz reference palette, dark surface (matches .streamlit/config.toml) ---
SURFACE = "#1a1a19"
INK_PRIMARY = "#ffffff"
INK_MUTED = "#c3c2b7"
GRIDLINE = "#2c2c2a"
SERIES = {"ACO": "#3987e5", "SA": "#d95926"}

st.set_page_config(page_title="TSP Solver — Live Demo", layout="wide")


def list_instances():
    return sorted(INST_DIR.glob("*.json"))


def solve_sa(instance: TSPInstance, seed: int, max_steps: int) -> dict:
    # Scale the cooling rate to the chosen step budget so the schedule always
    # finishes annealing (temperature down to ~0.2% of its start) within it.
    # A fixed rate only anneals fully at one particular budget, leaving the
    # tour visibly tangled at smaller ones.
    iters_per_temp = instance.n * 5
    num_coolings = max(1, max_steps // iters_per_temp)
    cooling_alpha = 0.002 ** (1 / num_coolings)

    cfg = SAConfig(
        init_accept_prob=0.8,
        uphill_samples=100,
        cooling_alpha=cooling_alpha,
        iters_per_temp=iters_per_temp,
        min_temp=1e-12,
        max_steps=max_steps,
        log_interval=max(1, max_steps // TARGET_FRAMES),
        seed=seed,
        use_step_budget_only=True,
        record_tour_history=True,
    )
    return sa_solve(instance, cfg)


def solve_aco(instance: TSPInstance, seed: int, iterations: int) -> dict:
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


def trim_after_convergence(tour_history, cost_history, hold: int = 6):
    """Cut the long flat tail once the best cost stops improving, keeping a
    short hold so the final reveal still registers."""
    last_improvement = 0
    for i in range(1, len(cost_history)):
        if cost_history[i][1] < cost_history[i - 1][1] - 1e-9:
            last_improvement = i
    cutoff = min(len(cost_history), last_improvement + hold + 1)
    return tour_history[:cutoff], cost_history[:cutoff]


def tour_xy(instance: TSPInstance, tour):
    xs = [instance.coords[c][0] for c in tour] + [instance.coords[tour[0]][0]]
    ys = [instance.coords[c][1] for c in tour] + [instance.coords[tour[0]][1]]
    return xs, ys


def build_figure(instance: TSPInstance, result: dict, algo: str):
    color = SERIES[algo]
    tour_history, cost_history = trim_after_convergence(result["tour_history"], result["history"])
    steps = [s for s, _ in cost_history]
    costs = [c for _, c in cost_history]
    xs_all = [c[0] for c in instance.coords]
    ys_all = [c[1] for c in instance.coords]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Tour", "Convergence"))

    # trace 0: cities (static, hoverable)
    fig.add_trace(
        go.Scatter(
            x=xs_all, y=ys_all, mode="markers",
            marker=dict(size=9, color=INK_PRIMARY, line=dict(width=1.5, color=SURFACE)),
            hovertemplate="city %{pointNumber}<br>x=%{x}, y=%{y}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )
    # trace 1: tour path (animated)
    tx0, ty0 = tour_xy(instance, tour_history[0][1])
    fig.add_trace(
        go.Scatter(x=tx0, y=ty0, mode="lines", line=dict(color=color, width=2.5), showlegend=False),
        row=1, col=1,
    )
    # trace 2: convergence line (animated, grows)
    fig.add_trace(
        go.Scatter(x=[steps[0]], y=[costs[0]], mode="lines", line=dict(color=color, width=2.5), showlegend=False),
        row=1, col=2,
    )
    # trace 3: convergence "you are here" dot (animated)
    fig.add_trace(
        go.Scatter(
            x=[steps[0]], y=[costs[0]], mode="markers",
            marker=dict(size=11, color=color, line=dict(width=2, color=SURFACE)),
            showlegend=False,
        ),
        row=1, col=2,
    )

    frames = []
    for i, (step, tour) in enumerate(tour_history):
        tx, ty = tour_xy(instance, tour)
        frames.append(go.Frame(
            name=str(i),
            data=[
                go.Scatter(x=tx, y=ty),
                go.Scatter(x=steps[: i + 1], y=costs[: i + 1]),
                go.Scatter(x=[steps[i]], y=[costs[i]]),
            ],
            traces=[1, 2, 3],
            # `title` is a single object, so each frame cleanly replaces it.
            # `annotations` is a list and Plotly merges rather than replaces
            # it per frame, which stacks every frame's text on top of the last.
            layout=go.Layout(title=dict(
                text=f"cost = {costs[i]:.2f}  (step {step})",
                x=0.22, xanchor="center", font=dict(color=INK_PRIMARY, size=16),
            )),
        ))
    fig.frames = frames

    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=INK_MUTED),
        margin=dict(t=90, b=60, l=50, r=20),
        height=520,
        title=dict(
            text=f"cost = {costs[0]:.2f}  (step {steps[0]})",
            x=0.22, xanchor="center", font=dict(color=INK_PRIMARY, size=16),
        ),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.99, y=1.28, xanchor="right",
            buttons=[
                dict(label="▶ Play", method="animate", args=[None, dict(
                    frame=dict(duration=140, redraw=True), fromcurrent=True, transition=dict(duration=0),
                )]),
                dict(label="⏸ Pause", method="animate", args=[[None], dict(
                    frame=dict(duration=0, redraw=False), mode="immediate",
                )]),
            ],
        )],
        sliders=[dict(
            active=0, y=-0.12, x=0.0, len=1.0,
            currentvalue=dict(prefix="step ", font=dict(color=INK_MUTED)),
            steps=[dict(
                method="animate", label=str(steps[i]),
                args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
            ) for i in range(len(tour_history))],
        )],
    )
    fig.update_xaxes(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE, row=1, col=1, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE, row=1, col=1)
    fig.update_xaxes(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE, row=1, col=2,
                      range=[min(steps), max(steps)], title_text="iteration / step")
    pad = (max(costs) - min(costs)) * 0.08 or 1.0
    fig.update_yaxes(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE, row=1, col=2,
                      range=[min(costs) - pad, max(costs) + pad], title_text="best cost so far")

    return fig, len(tour_history)


def main():
    st.title("TSP Solver — Live Demo")
    st.caption(
        "Solve a Traveling Salesman instance with Simulated Annealing or Ant "
        "Colony Optimization, then watch the tour untangle frame by frame."
    )

    instances = list_instances()
    if not instances:
        st.error(f"No instance files found in {INST_DIR}")
        return

    with st.sidebar:
        st.header("Configuration")
        inst_path = st.selectbox("Instance", instances, format_func=lambda p: p.name)
        algo = st.radio("Algorithm", ["ACO", "SA"])
        seed = st.number_input("Seed", value=42, step=1)

        if algo == "SA":
            budget = st.slider("Max steps", 5_000, 300_000, 100_000, step=5_000)
        else:
            budget = st.slider("Iterations", 20, 300, 150, step=10)

        solve_clicked = st.button("Solve", type="primary")

    if solve_clicked:
        instance = TSPInstance.load(str(inst_path))
        with st.spinner(f"Running {algo}..."):
            result = solve_sa(instance, seed, budget) if algo == "SA" else solve_aco(instance, seed, budget)
        st.session_state["instance"] = instance
        st.session_state["result"] = result
        st.session_state["algo"] = algo

    if "result" not in st.session_state:
        st.info("Pick an instance and algorithm, then click **Solve**.")
        return

    instance = st.session_state["instance"]
    result = st.session_state["result"]
    algo = st.session_state["algo"]

    fig, n_frames = build_figure(instance, result, algo)
    st.plotly_chart(fig, width="stretch", theme=None)

    col1, col2, col3 = st.columns(3)
    col1.metric("Best cost found", f"{result['best_cost']:.2f}")
    col2.metric("Cities", instance.n)
    col3.metric("Animation frames", n_frames)


if __name__ == "__main__":
    main()
