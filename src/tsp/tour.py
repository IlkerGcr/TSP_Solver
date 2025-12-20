from __future__ import annotations
from typing import List
from tsp.instance import TSPInstance

Tour = List[int]  # A tour is represented as a list of node indices

def tour_length(tour: Tour, instance: TSPInstance) -> float:
    n = instance.n
    if len(tour) != n:  # Tur uzunluğu, instance'daki düğüm sayısına eşit olmalı
        raise ValueError("Tour length does not match number of nodes in instance.")
    dist = instance.dist
    total = 0.0
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]  # Son şehirden ilk şehre dönüş ekleme 
        total += dist[a][b]
    return total


"""    
Calculate the total length of a given tour for a TSP instance.

    Args:
        tour (Tour): A list of node indices representing the tour.
        instance (TSPInstance): The TSP instance containing the distance matrix.
    Returns:
        float: The total length of the tour.

    Complexity: O(n), where n is the number of nodes(hospital, city) in the TSP instance.
"""