import random

from tsp.instance import TSPInstance
from tsp.exact import held_karp, brute_force_bnb


def test_held_karp_square_matches_known_optimal():
    inst = TSPInstance.from_coords([(0, 0), (0, 1), (1, 1), (1, 0)])
    _, cost = held_karp(inst, start=0)
    assert abs(cost - 4.0) < 1e-9


def test_held_karp_collinear_points_matches_known_optimal():
    inst = TSPInstance.from_coords([(0, 0), (1, 0), (2, 0)])
    _, cost = held_karp(inst, start=0)
    assert abs(cost - 4.0) < 1e-9


def test_brute_force_matches_held_karp_on_random_small_instance():
    rng = random.Random(11)
    coords = [(rng.random() * 100, rng.random() * 100) for _ in range(7)]
    inst = TSPInstance.from_coords(coords)

    _, hk_cost = held_karp(inst, start=0)
    _, bf_cost, timed_out = brute_force_bnb(inst, start=0)

    assert not timed_out
    assert abs(hk_cost - bf_cost) < 1e-6
