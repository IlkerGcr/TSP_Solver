from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import math
import random
from time import perf_counter

from tsp import instance
from tsp.instance import TSPInstance
from tsp.moves    import apply_two_opt, two_opt_delta

Tour = List[int]  


def tour_cost(instance: TSPInstance, tour: Tour) -> float:

    dist= instance.dist
    n= instance.n
    total = 0.0
    for k in range(n):
        a = tour[k]
        b = tour[(k+1) % n]
        total += dist[a][b]
    return total



@dataclass
class ACOConfig: # ACO PARAMETRELERİ
    num_ants: int = 0       # n <= 0 ( şehir sayısı kadar karınca )
    alpha: float = 1.0      # Pheromone etkisi
    beta : float = 2.0      # Heuristic etkisi
    rho  : float = 0.1      # Buharlaşma oranı
    q    : float = 1.0      # Pheromone sabiti 

    iterations: int = 100   # İterasyon sayısı
    time_limit_sec: Optional[float] = None         # Zaman limiti (saniye)

    #OPsiyonel yerel arama parametreleri
    use_local_search: bool = False
    local_search_passes: int = 1   # her tur için kaç tane 2-opt geçişi

    # Feromon clamp (çok küçülür yada büyürse diye sınır koyma)
    tau_min: Optional[float] = None
    tau_max: Optional[float] = None

    # Kayıt( tekrarlanabilirlik)
    log_interval: int = 1
    seed: Optional[int] = None


def build_heuristic(instance: TSPInstance) -> List[List[float]]:   
    # ETA MATRİSİ OLUŞTURMA
    #eta[i][j] = 1 / dist[i][j] (i != j), else 0 

    n= instance.n
    dist= instance.dist
    eta= [[0.0]* n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and dist[i][j] > 0.0:
                eta[i][j] = 1.0 / dist[i][j]
            else:
                eta[i][j] = 0.0
    return eta


def construct_tour(
        instance: TSPInstance,
        tau: List[List[float]],
        eta: List[List[float]],
        cfg: ACOConfig,
        rng: random.Random
) -> Tour:
    # Tek bir karınca için tur oluşturma
    n= instance.n
    start = rng.randrange(n)  # Rastgele başlangıç şehri
    tour= [start]
    visited= {start}

    for _ in range(n-1):
        current = tour[-1]
        scores: List[Tuple[int, float]] = []  # (şehir, skor)
        denom = 0.0

        for j in range(n):
            if j in visited:
                continue
            tij = tau[current][j]
            eij = eta[current][j]
            if tij <= 0.0 or eij <= 0.0:
                score = 0.0
            else:
                score = (tij ** cfg.alpha) * (eij ** cfg.beta)
            scores.append((j, score))
            denom += score

        if denom <= 0.0: # tüm skorlar sıfırsa rastgele seçim yap
            canditates = [j for j in range(n) if j not in visited]
            next_city = rng.choice(canditates)
        else: # Rulet seçimi
            r = rng.random() * denom
            cumulative = 0.0
            next_city = scores[-1][0]  # varsayılan son şehir
            for j, s in scores:
                cumulative += s
                if cumulative >= r:
                    next_city = j
                    break
        
        tour.append(next_city)
        visited.add(next_city)

    return tour



def two_opt_local_search_delta(
        instance: TSPInstance,
        tour: Tour,
        max_passes: int = 1
) -> Tuple[Tour, float]:
    # 2-opt yerel arama (delta hesaplamalı)
    n= instance.n
    best_tour= tour[:]
    best_cost= tour_cost(instance, best_tour)

    for _ in range(max_passes):
        improved= False
        for i in range(n-1):
            for j in range(i+2, n):
                if i == 0 and j == n - 1:
                    continue  # Geçersiz 2-opt
                delta = two_opt_delta(instance, best_tour, i, j)
                if delta < -1e-12:  # İyileşme varsa                                   **** PARAMETRE. TEKARARDAN BAK
                    best_tour = apply_two_opt(best_tour, i, j)
                    best_cost += delta
                    improved = True
        if not improved:
            break

    return best_tour, best_cost


def aco_solve(instance: TSPInstance, cfg: ACOConfig) -> Dict[str, Any]:
    # ACO ANA FONKSİYONU. Baseline ACO + opsiyonel 2-opt yerel arama


    rng= random.Random(cfg.seed)
    n= instance.n 

    num_ants= cfg.num_ants if cfg.num_ants > 0 else n

    #Feromon ve heuristic matrisleri
    tau0 = 1.0
    tau = [[tau0]* n for _ in range(n)]  # Başlangıç feromon değeri
    for i in range(n):
        tau[i][i] = 0.0

    
    eta = build_heuristic(instance)

    best_tour: Optional[Tour] = None
    best_cost = math.inf

    history: List[Tuple[int, float]] = []

    t_start = perf_counter()
    last_it = 0

    for it in range(1, cfg.iterations + 1):
        last_it = it

        # Zaman limiti varsa kontol et
        if cfg.time_limit_sec is not None:
            if perf_counter() - t_start > cfg.time_limit_sec:
                break

        ant_tours: List[Tour]  = []
        ant_costs: List[float] = []

        for _ in range(num_ants):
            tour = construct_tour(instance, tau, eta, cfg, rng)

            if cfg.use_local_search:
                tour, cost = two_opt_local_search_delta(
                    instance,
                    tour,
                    max_passes = cfg.local_search_passes
                )
            else:
                cost = tour_cost(instance, tour)

            ant_tours.append(tour)
            ant_costs.append(cost)

            if cost < best_cost:
                best_tour = tour
                best_cost = cost

        # BUHARLAŞMA
        rho = cfg.rho
        for i in range(n):
            for j in range(i+1, n):
                tau[i][j] *= (1.0 - rho)
                tau[j][i] = tau[i][j]

        # Tüm karıncalar feromon eklemesi
        q = cfg.q
        for tour, cost in zip(ant_tours, ant_costs):
            if cost <= 0.0:
                continue
            deposit = q / cost
            for k in range(n):
                a = tour[k]
                b = tour[(k+1) % n]
                tau[a][b] += deposit
                tau[b][a] += deposit

        # Feromon clamp(tau_min ve tau_max verilmiş ise)
        if cfg.tau_min is not None or cfg.tau_max is not None:
            tmin = cfg.tau_min if cfg.tau_min is not None else math.inf
            tmax = cfg.tau_max if cfg.tau_max is not None else -math.inf
            for i in range(n):
                for j in range(i + 1, n):
                    val = tau[i][j]
                    if val < tmin:
                        val = tmin
                    if val > tmax:
                        val = tmax
                    tau[i][j] = val
                    tau[j][i] = val

        # Kayıt
        if cfg.log_interval > 0 and it % cfg.log_interval == 0:
            history.append((it, best_cost))

    t_end = perf_counter()

    if best_tour is None:
        best_tour = list(range(n))
        best_cost = tour_cost(instance, best_tour)


    if not history or history[-1][0] != last_it:
        history.append((last_it, best_cost))

    return {
        "best_tour": best_tour,
        "best_cost": best_cost,
        "iterations_done": last_it,
        "runtime_sec": t_end - t_start,
        "history": history,
        "config": cfg,
    }