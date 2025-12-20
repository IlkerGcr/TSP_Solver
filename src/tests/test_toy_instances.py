from tsp.instance import TSPInstance
from tsp.tour import tour_length




def test_square_perimeter():
    inst = TSPInstance.from_coords([(0, 0), (0, 1), (1, 1), (1, 0)])
    tour = [0, 1, 2, 3]
    assert abs(tour_length(tour, inst) - 4.0) < 1e-9


def test_line_three_points():
    inst = TSPInstance.from_coords([(0, 0), (1, 0), (2, 0)])
    tour = [0, 1, 2]
    assert abs(tour_length(tour, inst) - 4.0) < 1e-9


"""   
 Test the tour_length function with simple TSP instances.

    Tests:
        - A square with perimeter 4.0
        - A line of three points with total length 4.0  
        Asserts that the computed tour lengths match expected values within a small tolerance.
        
   
"""