"""
AMAÇ: Bu dosya, TSP turunda 2-OPT hareketini uygulamak ve maliyet farkini hesaplamak için işlevler sağlar.

- apply_two_opt: Verilen bir turda 2-OPT hareketini uygulayarak yeni bir tur döndürür.
- two_opt_delta: 2-OPT hareketinin maliyet farkini O(1) zamanda hesaplar.
- sample_two_opt_indices: Geçerli (i, k) çiftlerini rastgele örnekler.
"""



from __future__ import annotations
from typing import List, Tuple
from tsp.instance import TSPInstance

Tour = List[int]

def apply_two_opt(tour: Tour, i: int, k: int) -> Tour:
    """
    Return a new tour obtained by applying a *** 2-OPT MOVE ***.
    Reverses the segment tour[i : k+1].
    """
    if i >= k:
        raise ValueError(f"Apply two opt expects i < k, got i={i}, k={k}")
        
    # Turun i..k arası segmentini ters çevirir. 2-opt’un standart operasyonu. O(n) çünkü liste dilimleme + reverse var.
    return tour[:i] + list(reversed(tour[i:k + 1])) + tour[k + 1:]



# 2-OPT hareketinin maliyet farkini O(1) zamanda hesaplar.
def two_opt_delta(instance: TSPInstance, tour: Tour, i: int, k: int) -> float:
    """
    O(1) cost difference of applying 2-opt.
    """
    n = instance.n
    
    # 1. Validate indices
    if not (0 <= i < k < n):
        raise ValueError(f"Invalid indices: i={i}, k={k}, n={n}")

    # 2. Handle full cycle reversal (no change in cost for TSP)
    if i == 0 and k == n - 1:
        return 0.0
    
    dist = instance.dist
    
    # 3. Identify the 4 nodes involved in the swap
    a, b = tour[(i - 1) % n], tour[i]
    c, d = tour[k], tour[(k + 1) % n]

    # 4. Calculate delta: (new edges) - (old edges)
    old_cost = dist[a][b] + dist[c][d]
    new_cost = dist[a][c] + dist[b][d]
    
    return new_cost - old_cost



# Rastgele geçerli (i, k) çiftlerini seçmek.
def sample_two_opt_indices(n: int, rng) -> Tuple[int, int]:
    """
    Sample a valid (i, k) pair for 2-opt.
    """
    while True:
        # Get two distinct, sorted numbers
        i, k = sorted(rng.sample(range(n), 2))

        # Reject adjacent edges (useless move). Yan yana olan nodeları kesmek gereksiz 
        if k == i + 1:
            continue
            
        # Reject full circle reversal (useless move). tüm turu ters çevirmekte gereksiz 
        if i == 0 and k == n - 1:
            continue
            
        return i, k