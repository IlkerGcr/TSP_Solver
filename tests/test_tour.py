import pytest

from tsp.instance import TSPInstance
from tsp.tour import tour_length


def test_square_perimeter():
    inst = TSPInstance.from_coords([(0, 0), (0, 1), (1, 1), (1, 0)])
    assert abs(tour_length([0, 1, 2, 3], inst) - 4.0) < 1e-9


def test_line_three_points():
    inst = TSPInstance.from_coords([(0, 0), (1, 0), (2, 0)])
    assert abs(tour_length([0, 1, 2], inst) - 4.0) < 1e-9


def test_wrong_length_tour_raises():
    inst = TSPInstance.from_coords([(0, 0), (1, 0), (2, 0)])
    with pytest.raises(ValueError):
        tour_length([0, 1], inst)
