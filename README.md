
## Opis projekta

Sistem za analitiku grafova velikih razmera razvijen za potrebe FastTransitNetwork (FTN) mreže. Implementira tri ključna algoritma za dijagnostiku kvarova i analizu mreže:

- **BFS (Breadth-First Search)** - Određivanje dostižnosti čvorova i najkraćih rastojanja
- **WCC (Weakly Connected Components)** - Identifikacija slabo povezanih komponenti
- **PageRank** - Procena važnosti čvorova u mreži

Svaki algoritam implementiran je u **sekvencijalnoj** i **paralelnoj** verziji sa detaljnom verifikacijom korektnosti i analizom performansi.

---

## Karakteristike

✅ **Efikasna CSR (Compressed Sparse Row) reprezentacija grafova**  
✅ **Bidirekciona CSR struktura za algoritme koji zahtevaju reverse edges**  
✅ **Sekvenencijalne i paralelne implementacije svih algoritama**  
✅ **CLI interfejs sa svim potrebnim opcijama**  
✅ **40+ unit testova za verifikaciju korektnosti**  
✅ **Python skripte za automatizovano benchmarking i vizualizaciju**  
✅ **Minimalan memorijski overhead (<1%) kod paralelnih verzija**  
✅ **Skalabilnost do 16 niti**  

---

## Tehnologije

- **Rust** 1.75+ (programski jezik)
- **Cargo** 1.75+ (build sistem)
- **Rayon** - biblioteka za paralelizaciju
- **Clap** - CLI argument parsing
- **Python 3.8+** - za benchmarking skripte
- **Matplotlib** - za vizualizaciju rezultata

---

## Struktura projekta
```
graph_algorithms/
├── src/
│   ├── main.rs              # CLI interfejs i entry point
│   ├── graph/
│   │   ├── mod.rs           # Graph modul
│   │   └── csr.rs           # CSR i BidirectionalCSR implementacija
│   └── algorithms/
│       ├── mod.rs           # Algorithms modul
│       ├── bfs.rs           # BFS sekvenencijalno i paralelno
│       ├── wcc.rs           # WCC sekvenencijalno i paralelno
│       └── pagerank.rs      # PageRank sekvenencijalno i paralelno
├── Cargo.toml               # Rust dependencies
├── benchmark.py             # Performance benchmark skripta
├── memory_benchmark.py      # Memory benchmark skripta
├── README.md                # Ovaj fajl
└── graph_analytics_report.pdf  # Detaljni izveštaj (LaTeX)
```

---

## Instalacija i build

### Preduslovi
```bash
# Provera Rust verzije
rustc --version  # Minimalno 1.75

# Provera Cargo verzije
cargo --version  # Minimalno 1.75
```

### Build
```bash
# Debug build
cargo build

# Release build (PREPORUČENO za benchmarking)
cargo build --release

# Pokretanje testova
cargo test

# Pokretanje testova sa detaljnim outputom
cargo test -- --nocapture

# Pokretanje testova u release modu
cargo test --release
```

---

## Korišćenje

### CLI interfejs

#### BFS (Breadth-First Search)
```bash
# Sekvencijalna verzija
./target/release/tool bfs --input edges.txt --source 0 --mode seq --out bfs_seq.txt

# Paralelna verzija sa 8 niti
./target/release/tool bfs --input edges.txt --source 0 --mode par --threads 8 --out bfs_par.txt
```

**Parametri:**
- `--input` - Ulazni fajl sa edge list-om
- `--source` - Izvorni čvor za BFS
- `--mode` - `seq` za sekvencijalno, `par` za paralelno
- `--threads` - Broj niti (samo za paralelni mod)
- `--out` - Izlazni fajl sa rezultatima

**Format izlaza:** Svaka linija sadrži `vertex distance`

#### WCC (Weakly Connected Components)
```bash
# Sekvencijalna verzija
./target/release/tool wcc --input edges.txt --mode seq --out wcc_seq.txt

# Paralelna verzija sa 8 niti
./target/release/tool wcc --input edges.txt --mode par --threads 8 --out wcc_par.txt
```

**Parametri:**
- `--input` - Ulazni fajl sa edge list-om
- `--mode` - `seq` ili `par`
- `--threads` - Broj niti (samo za paralelni mod)
- `--out` - Izlazni fajl

**Format izlaza:** Svaka linija sadrži `vertex component_id`

#### PageRank
```bash
# Sekvencijalna verzija
./target/release/tool pagerank --input edges.txt --mode seq --out pr_seq.txt \
  --alpha 0.85 --iters 100 --eps 1e-6

# Paralelna verzija sa 8 niti
./target/release/tool pagerank --input edges.txt --mode par --threads 8 --out pr_par.txt \
  --alpha 0.85 --iters 100 --eps 1e-6
```

**Parametri:**
- `--input` - Ulazni fajl sa edge list-om
- `--mode` - `seq` ili `par`
- `--threads` - Broj niti (samo za paralelni mod)
- `--out` - Izlazni fajl
- `--alpha` - Damping faktor (default: 0.85)
- `--iters` - Maksimalan broj iteracija (default: 100)
- `--eps` - Tolerancija konvergencije (default: 1e-6)

**Format izlaza:** Svaka linija sadrži `vertex rank_value`

---

## Format ulaznog fajla

Tekstualni fajl gde svaka linija predstavlja jednu ivicu:
```
# Primer: edges.txt
0 1
0 2
1 2
1 3
2 3
3 4
```

- Svaka linija: `source_vertex destination_vertex`
- Čvorovi su ненегативни цели бројеви
- Graf je podrazumevano **usmereni**
- Linije koje počinju sa `#` se ignorišu (komentari)
- Prazne linije se ignorišu

---

## Benchmarking

### Performance benchmark
```bash
# Instalacija Python zavisnosti
pip install matplotlib numpy

# Pokretanje benchmark-a
python3 benchmark.py
```

Skripta:
- Generiše male (1,000 čvorova) i velike (5,000,000 čvorova) grafove
- Izvršava sve algoritme sekvenencijalno i paralelno sa 1, 2, 4, 8 nitima
- Meri vreme izvršavanja i ubrzanje
- Generiše grafike i tabele sa rezultatima
- Čuva rezultate u `benchmark_results/` direktorijumu

### Memory benchmark
```bash
# Pokretanje memory benchmark-a
python3 memory_benchmark.py
```

Skripta:
- Generiše veliki graf (50,000 čvorova)
- Meri peak i average memorijsku potrošnju
- Poredi sekvenencijalne i paralelne verzije
- Generiše grafike memorijske potrošnje
- Radi samo na **Linux** i **macOS** sistemima

---
