#Generate and save random TSP instances to JSON files.


from pathlib import Path
import sys
import random


# Bu kısım proje dizinini sys.path'e ekler ki tsp.instance modülünü içe aktarabilelim
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsp.instance import TSPInstance

BASE_DIR = SRC_DIR.parent
DATA_DIR = BASE_DIR / "data" / "instances"





# Rastgele TSP örneği oluşturma fonksiyonu
def make_random_instance(n: int, seed: int, box_size: float = 1000.0) -> TSPInstance:
    rng = random.Random(seed)
    coords = [
        (rng.random() * box_size, rng.random() * box_size)
        for _ in range(n)
    ]
    return TSPInstance.from_coords(coords)

# Ana fonksiyon: çeşitli boyutlarda örnekler oluştur ve kaydet
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating instances in {DATA_DIR}")

    configs = [
        ("tiny",   10, [1, 2, 3]),
        ("tiny",   12, [1, 2, 3]),
        ("tiny",   14, [1, 2, 3]),
        ("small",  20, [1, 2, 3]),
        ("medium", 50, [1, 2, 3]),
        ("large", 200, [1, 2, 3]),
    ]

    for label, n, seeds in configs:
        for s in seeds:
            inst = make_random_instance(n, s, box_size=1000.0)
            fname = DATA_DIR / f"{label}_n{n}_seed{s}.json"
            inst.save(str(fname))                                  # str conversion for safety
            print(f"Saved: {fname} (size: {n})")

if __name__ == "__main__":
    main()
