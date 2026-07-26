import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Yolları ayarla
THIS_DIR = Path(__file__).resolve().parent # experiments/
SRC_DIR = THIS_DIR.parent                  # src/
PROJECT_ROOT = SRC_DIR.parent              # TSP_proje-main/

# Importlar için src'yi ekle
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsp.instance import TSPInstance

def find_instance_dir():
    """Dosyaların nerede olduğunu otomatik bulmaya çalışır"""
    possible_paths = [
        SRC_DIR / "data" / "instances",       
        PROJECT_ROOT / "data" / "instances",  
        Path("data/instances").resolve()      
    ]

    for path in possible_paths:
        if path.exists():
            files = list(path.glob("*.json"))
            if files:
                print(f"✅ Dosyalar bulundu: {path}")
                return path, files
    
    return None, []

def generate_maps():
    # 1. Dosyaları bul
    instance_dir, json_files = find_instance_dir()
    
    if not instance_dir:
        print("\n❌ HATA: 'instances' klasörü veya .json dosyaları bulunamadı!")
        return

    # 2. Kayıt yerini ayarla
    MAPS_DIR = instance_dir.parent / "maps"
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Hedef Klasör: {MAPS_DIR}")
    print(f"🎨 Toplam {len(json_files)} harita çiziliyor...")

    # 3. Çizim Döngüsü
    for i, json_path in enumerate(json_files, 1):
        try:
            inst = TSPInstance.load(str(json_path))
            
            plt.figure(figsize=(8, 8))
            
            # --- DÜZELTME BURADA ---
            # 'points' yerine 'coords' kullanıyoruz.
            # Ayrıca liste mi yoksa numpy dizisi mi olduğunu dert etmemek için zip(*) kullanıyoruz.
            if hasattr(inst, 'coords'):
                points_data = inst.coords
            elif hasattr(inst, '_coords'):
                points_data = inst._coords
            else:
                # Eğer sınıf içinde attribute yoksa JSON'dan direkt okumayı deneyelim (fallback)
                import json
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    points_data = data.get('coords', [])

            if not points_data:
                print(f"⚠️ Uyarı: {json_path.name} içinde koordinat bulunamadı.")
                plt.close()
                continue

            # X ve Y'leri ayır
            xs, ys = zip(*points_data)
            
            # Çizim
            plt.scatter(xs, ys, c='red', marker='o', s=50, edgecolors='black', label='Şehirler')
            
            # Numaraları yaz
            for city_idx, (x, y) in enumerate(points_data):
                plt.annotate(str(city_idx), (x, y), xytext=(4, 4), 
                             textcoords='offset points', fontsize=9, fontweight='bold')

            plt.title(f"Harita: {json_path.stem}\n(N={len(points_data)})")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.axis('equal')
            
            # Kaydet
            output_filename = f"{json_path.stem}.png"
            plt.savefig(MAPS_DIR / output_filename, dpi=100)
            plt.close()
            
            print(f"[{i}/{len(json_files)}] OK -> {output_filename}")
            
        except Exception as e:
            print(f"⚠️ Hata ({json_path.name}): {e}")
            import traceback
            traceback.print_exc()

    print("\n✅ İşlem tamamlandı! Görseller 'maps' klasörüne kaydedildi.")

if __name__ == "__main__":
    generate_maps()