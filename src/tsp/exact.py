from __future__ import annotations
from typing import List, Tuple, Optional
import math
from time import perf_counter

from tsp.instance import TSPInstance

Tour = List[int]


def held_karp(instance: TSPInstance, start: int = 0) -> Tuple[Tour, float]:
    """
    Exact TSP via Held–Karp DP. Intended for small n (10/12/14).
    Returns (tour, cost). Tour starts at `start`.
    """
    n = instance.n
    dist = instance.dist
    if n <= 1:
        return [0], 0.0
    if not (0 <= start < n):
        raise ValueError("start out of range")

    nodes = [i for i in range(n) if i != start]
    m = len(nodes)
    idx_of = {node: idx for idx, node in enumerate(nodes)}

    # dp[mask][j] implemented as dict for simplicity
    dp = {}
    parent = {}

    for j in nodes:
        mask = 1 << idx_of[j]
        dp[(mask, j)] = dist[start][j]
        parent[(mask, j)] = start

    for mask in range(1, 1 << m):
        for j in nodes:
            jbit = 1 << idx_of[j]
            if not (mask & jbit):
                continue
            prev_mask = mask ^ jbit
            if prev_mask == 0:
                continue

            best_val = math.inf
            best_prev = -1
            for i in nodes:
                ibit = 1 << idx_of[i]
                if not (prev_mask & ibit):
                    continue
                val = dp[(prev_mask, i)] + dist[i][j]
                if val < best_val:
                    best_val = val
                    best_prev = i

            dp[(mask, j)] = best_val
            parent[(mask, j)] = best_prev

    full_mask = (1 << m) - 1

    best_cost = math.inf
    best_end = -1
    for j in nodes:
        val = dp[(full_mask, j)] + dist[j][start]
        if val < best_cost:
            best_cost = val
            best_end = j

    # Reconstruct
    rev = [best_end]
    mask = full_mask
    j = best_end
    while True:
        pj = parent[(mask, j)]
        if pj == start:
            break
        mask ^= (1 << idx_of[j])
        j = pj
        rev.append(j)

    rev.reverse()
    tour = [start] + rev
    return tour, best_cost


def _nearest_neighbor_upper_bound(instance: TSPInstance, start: int = 0) -> float:
    """Quick greedy tour to get an initial upper bound for branch-and-bound."""
    n = instance.n
    dist = instance.dist
    unvis = set(range(n))
    unvis.remove(start)
    cur = start
    cost = 0.0
    while unvis:
        nxt = min(unvis, key=lambda j: dist[cur][j])
        cost += dist[cur][nxt]
        cur = nxt
        unvis.remove(nxt)
    cost += dist[cur][start]
    return cost


def brute_force_bnb(
    instance: TSPInstance,
    start: int = 0,
    time_limit_sec: Optional[float] = None,
) -> Tuple[Optional[Tour], float, bool]:
    """
    Brute force with simple Branch & Bound.
    Returns: (best_tour or None if timeout, best_cost, timed_out)
    Note: Still factorial; n=10 ok, n=12 maybe, n=14 usually timeout.
    """
    n = instance.n
    dist = instance.dist
    t0 = perf_counter()

    if n <= 1:
        return [0], 0.0, False

    # Fix start to remove rotational symmetry
    nodes = [i for i in range(n) if i != start]

    # Extra symmetry reduction: force second city < last city
    # This removes reverse-duplicate tours in symmetric TSP.
    # We'll enforce this only when we complete a tour.
    best_cost = _nearest_neighbor_upper_bound(instance, start)
    best_tour: Optional[Tour] = None
    timed_out = False

    path = [start]
    used = [False] * n
    used[start] = True

    def rec(cur: int, cost_so_far: float) -> None:
        nonlocal best_cost, best_tour, timed_out

        if time_limit_sec is not None and (perf_counter() - t0) > time_limit_sec:
            timed_out = True
            return

        # Bound: if already worse than best, prune
        if cost_so_far >= best_cost:
            return

        if len(path) == n:
            # close tour
            total = cost_so_far + dist[cur][start]

            # symmetry: second < last
            if n >= 3 and not (path[1] < path[-1]):
                return

            if total < best_cost:
                best_cost = total
                best_tour = path[:]  # copy
            return

        # Try next nodes (simple ordering: nearest first helps pruning)
        candidates = [j for j in nodes if not used[j]]
        candidates.sort(key=lambda j: dist[cur][j])

        for nxt in candidates:
            used[nxt] = True
            path.append(nxt)
            rec(nxt, cost_so_far + dist[cur][nxt])
            path.pop()
            used[nxt] = False

            if timed_out:
                return

    rec(start, 0.0)
    return best_tour, best_cost, timed_out
