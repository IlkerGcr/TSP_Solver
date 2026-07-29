"""
Streamlit demo for the TSP solver suite.

Run locally with:
    pip install -e ".[demo]"
    streamlit run app/streamlit_app.py

Solves the selected instance once (SA or ACO already log a best-cost history
as they run), then lets you replay that convergence history as an animation
next to the final tour — no live callback wiring into the solvers needed.
"""
from __future__ import annotations
from pathlib import Path
import sys
import time

import streamlit as st
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
BASE_DIR = THIS_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsp.instance import TSPInstance
from solvers.sa import SAConfig, sa_solve
from solvers.aco import ACOConfig, aco_solve

INST_DIR = BASE_DIR / "data" / "instances"

st.set_page_config(page_title="TSP Solver Demo", layout="wide")


def list_instances():
    return sorted(INST_DIR.glob("*.json"))


def solve_sa(instance: TSPInstance, seed: int, max_steps: int) -> dict:
    cfg = SAConfig(
        init_accept_prob=0.8,
        uphill_samples=100,
        cooling_alpha=0.995,
        iters_per_temp=instance.n * 5,
        min_temp=1e-12,
        max_steps=max_steps,
        log_interval=max(1, max_steps // 200),
        seed=seed,
        use_step_budget_only=True,
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
        log_interval=1,
        seed=seed,
    )
    return aco_solve(instance, cfg)


def draw_tour(instance: TSPInstance, tour, cost: float):
    fig, ax = plt.subplots(figsize=(6, 6))
    xs = [instance.coords[c][0] for c in tour] + [instance.coords[tour[0]][0]]
    ys = [instance.coords[c][1] for c in tour] + [instance.coords[tour[0]][1]]
    ax.plot(xs, ys, "-", color="steelblue", linewidth=1.2, zorder=1)
    ax.scatter(
        [c[0] for c in instance.coords], [c[1] for c in instance.coords],
        color="crimson", edgecolors="black", zorder=2, s=40,
    )
    ax.set_title(f"cost = {cost:.2f}  (n={instance.n})")
    ax.axis("equal")
    ax.grid(True, linestyle=":", alpha=0.5)
    return fig


def draw_convergence(history, upto_index: int):
    fig, ax = plt.subplots(figsize=(6, 6))
    steps, costs = zip(*history[: upto_index + 1])
    ax.plot(steps, costs, color="darkorange")
    ax.set_xlim(history[0][0], history[-1][0])
    ax.set_ylim(min(c for _, c in history) * 0.98, max(c for _, c in history) * 1.02)
    ax.set_xlabel("iteration / step")
    ax.set_ylabel("best cost so far")
    ax.grid(True, linestyle=":", alpha=0.5)
    return fig


def main():
    st.title("TSP Solver — Live Demo")
    st.caption(
        "Solve a Traveling Salesman instance with Simulated Annealing or Ant "
        "Colony Optimization, then replay how the solution converged."
    )

    instances = list_instances()
    if not instances:
        st.error(f"No instance files found in {INST_DIR}")
        return

    with st.sidebar:
        st.header("Configuration")
        inst_path = st.selectbox(
            "Instance", instances, format_func=lambda p: p.name,
        )
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
            if algo == "SA":
                result = solve_sa(instance, seed, budget)
            else:
                result = solve_aco(instance, seed, budget)
        st.session_state["instance"] = instance
        st.session_state["result"] = result
        st.session_state["algo"] = algo
        st.session_state["anim_index"] = len(result["history"]) - 1

    if "result" not in st.session_state:
        st.info("Pick an instance and algorithm, then click **Solve**.")
        return

    instance = st.session_state["instance"]
    result = st.session_state["result"]
    history = result["history"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Final tour")
        st.pyplot(draw_tour(instance, result["best_tour"], result["best_cost"]))

    with col2:
        st.subheader("Convergence")
        play = st.checkbox("Auto-play convergence animation")
        anim_index = st.slider(
            "History point", 0, len(history) - 1,
            st.session_state.get("anim_index", len(history) - 1),
            key="anim_slider",
        )
        placeholder = st.empty()
        placeholder.pyplot(draw_convergence(history, anim_index))

        if play:
            for i in range(len(history)):
                placeholder.pyplot(draw_convergence(history, i))
                time.sleep(0.03)

    st.metric("Best cost found", f"{result['best_cost']:.2f}")


if __name__ == "__main__":
    main()
