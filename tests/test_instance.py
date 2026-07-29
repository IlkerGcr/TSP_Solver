import json
import math

from tsp.instance import TSPInstance


def test_from_coords_builds_symmetric_euclidean_matrix():
    inst = TSPInstance.from_coords([(0, 0), (3, 4), (3, 0)])
    assert inst.n == 3
    assert inst.dist[0][1] == inst.dist[1][0] == 5.0
    assert inst.dist[0][0] == 0.0
    assert inst.dist[1][2] == math.hypot(0, 4)


def test_json_round_trip_preserves_coords():
    inst = TSPInstance.from_coords([(1.5, 2.5), (0.0, 0.0), (9.0, -3.0)])
    restored = TSPInstance.from_json(inst.to_json())
    assert restored.coords == inst.coords
    assert restored.dist == inst.dist


def test_save_load_round_trip(tmp_path):
    inst = TSPInstance.from_coords([(0, 0), (1, 1), (2, 2), (3, 0)])
    path = tmp_path / "instance.json"
    inst.save(str(path))

    loaded = TSPInstance.load(str(path))
    assert loaded.n == inst.n
    assert loaded.coords == inst.coords

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert "coords" in payload
