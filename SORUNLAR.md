# TSP_proje — Bulunan Sorunlar / Kritik Geliştirmeler

1. **`.gitignore` işlevsiz.** Şu an içinde sadece anlamsız bir satır var (`IVIRZIVIR/`). Bu yüzden derlenmiş Python dosyaları repoya commit edilmiş (`src/solvers/__pycache__/aco.cpython-313.pyc`, `src/tsp/__pycache__/*.pyc` gibi). Standart bir Python `.gitignore` eklenip bu dosyaların repodan temizlenmesi gerekiyor:
   ```bash
   git rm -r --cached **/__pycache__
   ```
2. **README yoktu** (bu dosyayla eklendi) — proje aslında oldukça gelişmiş (3 farklı algoritma + paralel grid search) ama dışarıdan bakan biri bunu anlayamıyordu. Bu, GitHub'daki en güçlü projelerinden biri olabilir.
3. **`rng.py` içinde erişilemez docstring.** `make_rng` fonksiyonunda docstring, `return` satırından SONRA yazılmış — hiçbir zaman çalışmaz/görüntülenmez. Docstring'in fonksiyonun en başına taşınması gerekiyor.
4. **`run_grid_search.py` dosya başlığı tutarsız.** Dosyanın en üstünde yorum olarak `# src/experiments/run_grid_search_v3.py` yazıyor ama gerçek dosya adı `run_grid_search.py` — muhtemelen eski bir dosya adından kalma, kafa karıştırıcı.
5. **`run_large_benchmark_new.py` dosya adında "_new" var.** Versiyonlama dosya adında değil, git commit geçmişinde tutulmalı. İsim `run_large_benchmark.py` olarak sadeleştirilebilir.
6. **`data/results/` içinde büyük CSV/JSONL dosyaları commit edilmiş.** Bu tür üretilmiş/tekrar oluşturulabilir çıktı dosyaları genelde repoya değil `.gitignore`'a eklenir, repo boyutunu gereksiz şişiriyor.

## Öncelik
Bu proje CV'nin en güçlü kartlarından biri olabilir — 1 ve 2 numaralı maddeler (temiz .gitignore + README, ki README eklendi) en yüksek öncelikli. Diğerleri kozmetik.
