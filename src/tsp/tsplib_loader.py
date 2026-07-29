"""
Minimal parser for the TSPLIB `.tsp` format (EUC_2D instances only).

TSPLIB is the standard benchmark set for the Traveling Salesman Problem;
using it lets solver results be compared against externally published,
independently verifiable optimal tour lengths instead of only against
this project's own randomly generated instances.
"""
from __future__ import annotations
from typing import List, Tuple

from tsp.instance import TSPInstance

Coord = Tuple[float, float]


def parse_tsplib_coords(text: str) -> List[Coord]:
    """Extract (x, y) coordinates from a TSPLIB EUC_2D NODE_COORD_SECTION."""
    edge_weight_type = None
    for line in text.splitlines():
        if line.strip().upper().startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":", 1)[1].strip()
            break

    if edge_weight_type is not None and edge_weight_type != "EUC_2D":
        raise ValueError(
            f"Unsupported EDGE_WEIGHT_TYPE '{edge_weight_type}'; only EUC_2D is supported."
        )

    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip().upper() == "NODE_COORD_SECTION":
            start = idx + 1
            break
    if start is None:
        raise ValueError("NODE_COORD_SECTION not found in TSPLIB file.")

    coords: List[Coord] = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped or stripped.upper() in ("EOF", "DEPOT_SECTION"):
            break
        parts = stripped.split()
        # TSPLIB rows are: "<index> <x> <y>"
        x, y = float(parts[1]), float(parts[2])
        coords.append((x, y))

    return coords


def load_tsplib_instance(path: str) -> TSPInstance:
    """Load a TSPLIB `.tsp` file into a TSPInstance."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    coords = parse_tsplib_coords(text)
    return TSPInstance.from_coords(coords)
