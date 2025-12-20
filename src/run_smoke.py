from tsp.instance import TSPInstance
from tsp.tour import tour_length
from tsp.moves import apply_two_opt, two_opt_delta


if __name__ == "__main__":
    inst = TSPInstance.from_coords([(0, 0), (0, 1), (1, 1), (1, 0)])
    print ("n = ", inst.n)
    print("tour length = ", tour_length([0, 1, 2, 3], inst))

    tour = [0, 1, 2, 3]
    i , k = 0 , 2

    new_tour = apply_two_opt(tour, i, k)
    print("delta = ", two_opt_delta(inst, tour, i, k))
    print("new lenght = ", tour_length(new_tour, inst))


"""Run a simple smoke test for TSPInstance and tour_length.

    This script creates a simple TSP instance (a square) and computes the length
    of a tour that visits all four corners in order. It prints the number of nodes
    and the computed tour length to verify basic functionality.


    
"""