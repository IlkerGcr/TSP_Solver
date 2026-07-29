import random

from tsp.instance import TSPInstance
from tsp.tour import tour_length
from tsp.moves import apply_two_opt, two_opt_delta, sample_two_opt_indices


def test_two_opt_delta_matches_full_recompute_square():
    inst = TSPInstance.from_coords([(0, 0), (0, 1), (1, 1), (1, 0)])
    tour = [0, 1, 2, 3]
    base = tour_length(tour, inst)

    for (i, k) in [(1, 2), (1, 3), (0, 2)]:
        if i == 0 and k == inst.n - 1:
            continue
        new_tour = apply_two_opt(tour, i, k)
        full = tour_length(new_tour, inst)
        delta = two_opt_delta(inst, tour, i, k)
        assert abs((base + delta) - full) < 1e-9


def test_two_opt_delta_full_reversal_is_zero_cost():
    inst = TSPInstance.from_coords([(0, 0), (0, 1), (1, 1), (1, 0)])
    tour = [0, 1, 2, 3]
    assert abs(two_opt_delta(inst, tour, 0, inst.n - 1) - 0.0) < 1e-12


def test_sample_two_opt_indices_never_returns_useless_moves():
    n = 8
    rng = random.Random(7)
    for _ in range(200):
        i, k = sample_two_opt_indices(n, rng)
        assert 0 <= i < k < n
        assert not (i == 0 and k == n - 1)
        assert k != i + 1
