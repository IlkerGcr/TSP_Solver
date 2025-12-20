"""
This file creates the TSPInstance class.       **  ŞEHİR NESNESİ (TSPInstance) SINIFI. **
creating them from locations and saving or loading them with JSON
"""


from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import math
import json

Coord = Tuple[float, float]  # ŞEHİR LİSTESİ (ÖRNEK: (x,y) KOORDİNATLARI)

@dataclass(frozen=True)     # this line means the class is constant. Cannot modify after creation
class TSPInstance:
    coords: List[Coord]
    dist: List[List[float]]  # Distance matrix: dist[i][j]   

    @property   # this line means n is a read-only property
    def n(self) -> int:    # n = ŞEHİR SAYISI
        return len(self.coords)
    
    @staticmethod  # staicmethod = this method belongs to the class, not to an instance

    # Asıl Şehirleri kurma kısmı. 
    def from_coords(coords: List[Coord]) -> "TSPInstance":  # Compute distance matrix from coordinates. 
        n = len(coords)
        dist = [[0.0] * n for _ in range(n)]

        for i in range(n):
            xi , yi = coords[i]
            for j in range(i + 1, n):
                xj, yj = coords[j]
                d = math.hypot(xi - xj, yi - yj)  # ÖKLİD MESAFE HESABI
                dist[i][j] = d
                dist[j][i] = d
        return TSPInstance(coords=coords, dist=dist)

    def to_json(self) -> str:   # Serialize TSPInstance to JSON .  ** to ** json
        payload = {"coords": self.coords}
        return json.dumps(payload)

    @staticmethod  
    def from_json(s: str) -> "TSPInstance": # Deserialize TSPInstance from JSON.   ** from ** json
        payload = json.loads(s)
        coords = [tuple(coord) for coord in payload["coords"]]
        return TSPInstance.from_coords(coords)   # Create instance from coordinates
    def save(self, path: str) -> None:   # Save TSPInstance to a file in JSON format
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @staticmethod
    def load(path: str) -> "TSPInstance":  # Load TSPInstance from a JSON file
        with open(path, "r", encoding="utf-8") as f:
            s = f.read()
        return TSPInstance.from_json(s)