# src/solvers/sa.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import math
import random
from time import perf_counter

from tsp.instance import TSPInstance
from tsp.moves import apply_two_opt, two_opt_delta, sample_two_opt_indices

Tour = List[int]


def tour_cost(instance: TSPInstance, tour: Tour) -> float:
    """Compute the total cost of the given tour (wrap-around)."""
    dist = instance.dist
    n = instance.n
    total = 0.0
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        total += dist[a][b]
    return total


@dataclass
class SAConfig:
    # SA PARAMETRELERİ
    init_accept_prob: float = 0.8
    uphill_samples: int = 100

    cooling_alpha: float = 0.995
    iters_per_temp: int = 500

    min_temp: float = 1e-3
    max_steps: int = 50_000

    log_interval: int = 100
    seed: Optional[int] = None

    # Tuning gibi durumlarda "eşit bütçe" için sadece max_steps ile durdur
    use_step_budget_only: bool = False

    # Kapalı: mevcut benchmark/tuning koşularının performansını etkilemez.
    # Açıkken, history ile aynı noktalarda best_tour anlık görüntüsü de kaydedilir
    # (görselleştirme/animasyon için).
    record_tour_history: bool = False


def estimate_initial_temp(
    instance: TSPInstance,
    tour: Tour,
    cfg: SAConfig,
    rng: random.Random
) -> float:
    """Başlangıç sıcaklığını (T0) instance ölçeğine göre tahmin et."""
    deltas: List[float] = []
    n = instance.n

    for _ in range(cfg.uphill_samples):
        i, k = sample_two_opt_indices(n, rng)
        delta = two_opt_delta(instance, tour, i, k)
        if delta > 0:
            deltas.append(delta)

    if not deltas:
        return 1.0

    avg_delta = sum(deltas) / len(deltas)
    p = cfg.init_accept_prob

    if p <= 0.0 or p >= 1.0:
        raise ValueError("Initial acceptance probability must be in (0,1)")

    return -avg_delta / math.log(p)


def sa_solve(instance: TSPInstance, cfg: SAConfig) -> Dict[str, Any]:
    """Simulated Annealing solver for TSP."""
    rng = random.Random(cfg.seed)

    tour: Tour = list(range(instance.n))
    rng.shuffle(tour)

    current_cost = tour_cost(instance, tour)
    best_tour = tour[:]
    best_cost = current_cost

    T = estimate_initial_temp(instance, tour, cfg, rng)

    history: List[Tuple[int, float]] = [(0, best_cost)]
    tour_history: List[Tuple[int, Tour]] = [(0, best_tour[:])] if cfg.record_tour_history else []
    step = 0
    t_start = perf_counter()

    n = instance.n

    # Eğer use_step_budget_only=True ise min_temp'i görmezden gel, sadece step bütçesine göre çalış
    while step < cfg.max_steps and (cfg.use_step_budget_only or T > cfg.min_temp):
        for _ in range(cfg.iters_per_temp):
            step += 1

            i, k = sample_two_opt_indices(n, rng)
            delta = two_opt_delta(instance, tour, i, k)

            if delta <= 0 or rng.random() < math.exp(-delta / T):
                tour = apply_two_opt(tour, i, k)
                current_cost += delta

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_tour = tour[:]

            if step % cfg.log_interval == 0:
                history.append((step, best_cost))
                if cfg.record_tour_history:
                    tour_history.append((step, best_tour[:]))

            if step >= cfg.max_steps:
                break

        T *= cfg.cooling_alpha

    t_end = perf_counter()

    if history[-1][0] != step:
        history.append((step, best_cost))
        if cfg.record_tour_history:
            tour_history.append((step, best_tour[:]))

    return {
        "best_tour": best_tour,
        "best_cost": best_cost,
        "steps": step,
        "runtime_sec": t_end - t_start,
        "history": history,
        "tour_history": tour_history,
        "config": cfg,
    }
