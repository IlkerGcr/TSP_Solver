from tsp.instance import TSPInstance
from tsp.tour import tour_length
from tsp.moves import apply_two_opt, two_opt_delta



def test_two_opt_delta_matches_full_recompute_square():
    inst = TSPInstance.from_coords([(0, 0), (0, 1), (1, 1), (1, 0)])
    tour = [0, 1, 2, 3]
    base = tour_length(tour, inst)


    # Try a few valid (i,k) pairs
    for (i,k) in [(1,2),(1,3),(0,2)]:
        if i == 0 and k == inst.n - 1:
            continue # skip full reversal
        new_tour = apply_two_opt(tour, i, k)
        full = tour_length(new_tour, inst)
        delta = two_opt_delta(inst, tour, i, k)
        assert abs((base + delta) - full) < 1e-9

"""
 Test that the two_opt_delta function matches the difference in tour lengths
 computed by a full recomputation after applying a 2-opt move.

    The test uses a square TSP instance and checks several (i,k) pairs for 2-opt moves.
    It asserts that the length computed using the delta matches the length from a full recompute
    within a small tolerance.
"""

def test_wraparound_indices_handled():
    inst = TSPInstance.from_coords([(0, 0), (0, 1), (1, 1), (1, 0)])
    tour = [0, 1, 2, 3, 4]
    base = tour_length(tour, inst)

    # Test wrap-around case
    i, k = 0 , 2  
    assert abs(two_opt_delta(inst, tour, 0, inst.n -1 ) - 0.0) < 1e-12


#  Test that two_opt_delta correctly handles wrap-around indices.