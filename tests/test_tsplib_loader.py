import pytest

from tsp.tsplib_loader import parse_tsplib_coords, load_tsplib_instance

SAMPLE_TSP = """NAME: sample4
TYPE: TSP
DIMENSION: 4
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0.0 0.0
2 0.0 1.0
3 1.0 1.0
4 1.0 0.0
EOF
"""


def test_parse_tsplib_coords_reads_node_coord_section():
    coords = parse_tsplib_coords(SAMPLE_TSP)
    assert coords == [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]


def test_parse_tsplib_coords_rejects_non_euc_2d():
    bad = SAMPLE_TSP.replace("EUC_2D", "GEO")
    with pytest.raises(ValueError):
        parse_tsplib_coords(bad)


def test_load_tsplib_instance_from_file(tmp_path):
    path = tmp_path / "sample4.tsp"
    path.write_text(SAMPLE_TSP, encoding="utf-8")

    inst = load_tsplib_instance(str(path))
    assert inst.n == 4
    assert inst.dist[0][1] == 1.0
