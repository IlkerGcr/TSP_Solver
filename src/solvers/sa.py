from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple,Dict,Any
import math
import random
from time import perf_counter
from tsp.instance import TSPInstance
from tsp.moves import apply_two_opt, two_opt_delta, sample_two_opt_indices

Tour = List[int]


def tour_cost(instance: TSPInstance, tour: Tour) -> float:
    """Compute the total cost of the given tour."""
    dist = instance.dist
    n = instance.n
    total = 0.0
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]  # wrap around to the start
        total += dist[a][b]
    return total

@dataclass    # ***** BU KISIM DEĞİŞMEZ. SA İÇİN OLAN PARAMETRELER BURADA TANIMLANIR.*****
class SAConfig: 
    init_accept_prob: float = 0.8  # initial acceptance probability for uphill moves  Başlangıçta %80 Hata Payı 
    uphill_samples: int = 100 # how many random moves to estimate avg /_\ (üçgen sembolü)

    cooling_alpha: float = 0.995  # cooling rate
    iters_per_temp: int = 500   # iterations per temperature level

    min_temp: float = 1e-3   # minimum temperature to stop
    max_steps : int = 50_000  # maximum number of steps

    log_interval: int = 100   # log history every this many steps
    seed: Optional[int] = None   


# Estimate initial temperature T_0
def estimate_initial_temp(  
        instance: TSPInstance,
        tour: Tour,
        cfg: SAConfig,
        rng: random.Random
) -> float:
    deltas: List[float] = []
    n = instance.n


    for _ in range(cfg.uphill_samples):  # Sample random 2-opt moves
        i, k = sample_two_opt_indices(n, rng)
        delta = two_opt_delta(instance, tour, i, k)
        if delta > 0:
            deltas.append(delta)

    if not deltas:
        return 1.0  # No uphill moves found; arbitrary small temp
    
    avg_delta = sum(deltas) / len(deltas)
    p = cfg.init_accept_prob   # p factor for T_0 calculation (Yukarıda belirtilmiş init_accept_prob)

    if p <= 0.0 or p >= 1.0:  # If degenarerate case -> use avg_delta as a rough scale
        raise ValueError("Initial acceptance probability must be in (0,1)")
        return avg_delta
    
    return -avg_delta / math.log(p)  # Solve for T in exp(-avg_delta / T) = p  or T = -avg_delta / ln(p). p genelde 0.8 alıyorlar.


def sa_solve(instance: TSPInstance, cfg: SAConfig) -> Dict[str, Any]:
    # Simulated Annealing solver for TSP problem. 

    rng = random.Random(cfg.seed)

    tour: Tour = list(range(instance.n))  # Initial tour: 0,1,2,...,n-1
    rng.shuffle(tour) # Randomize initial tour

    current_cost = tour_cost(instance, tour) 
    best_tour = tour[:] 
    best_cost = current_cost


    T = estimate_initial_temp(instance, tour, cfg, rng) # Initial temperature


    history : List[Tuple[int, float]] = [(0, best_cost)]  # (step, best_cost)
    step = 0

    t_start = perf_counter()

    n= instance.n

    while T > cfg.min_temp and step < cfg.max_steps:   # "Min temperature or max steps" SA DUR KISMI 
        for _ in range(cfg.iters_per_temp):

            step += 1   
            i,k = sample_two_opt_indices(n, rng)
            delta = two_opt_delta(instance, tour, i, k)

            if delta <= 0 or rng.random() < math.exp(-delta / T):
                # Accept move
                tour = apply_two_opt(tour, i, k)
                current_cost += delta

                # Update best solution found
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_tour = tour[:]

            #Log history periodically
            if step % cfg.log_interval == 0:   # (log_interval =100)
                history.append((step, best_cost))  

            if step >= cfg.max_steps:
                break
        
        T *= cfg.cooling_alpha   # Cool down temperature(TEMP DÜŞÜRME KISMI)
    t_end = perf_counter()

    if history[-1][0] != step:  # Ensure final step is logged
        history.append((step, best_cost))   # Eğer son adım mod != 0 ise, step loglanmamışsa logla
    

    
    return { "best_tour": best_tour, "best_cost": best_cost, "steps": step, 
            "runtime_sec": t_end - t_start, "history": history, "config": cfg
    }
