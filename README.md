# Enhanced LTL Model Checking for BPMN Collaboration Models with Graph Analysis

This repository contains a Python tool for checking **deadlock patterns** in BPMN 2.0 collaboration diagrams using **Linear Temporal Logic (LTL)** and a **graph-based analysis**.

The tool:

- Parses BPMN 2.0 XML (`.bpmn`) collaboration diagrams  
- Builds a **graph** and a **Kripke structure** (finite-trace semantics / LTLf)  
- Evaluates a small catalogue of **LTL properties**:
  - loop deadlock  
  - source deadlock  
  - improper structuring deadlock  
- Exports results to:
  - **terminal report** (human-readable)
  - **JSON** (machine-readable)
  - **Cypher** scripts for **Neo4j** visualisation

---

## 1. Requirements

- Python **3.10+**
- Standard Python libraries (no external dependencies for the basic checker)
- Optional (for visualisation):
  - Neo4j Desktop atau Neo4j Aura
  - Neo4j Browser untuk melihat graph

Contoh struktur folder:

    .
    ├── bpmn_ltl_checker.py
    ├── props.ltl
    ├── diagram/
    │   ├── case simple deadlock.bpmn
    │   ├── StructureDeadlock.bpmn
    │   └── ...
    └── README.md

## 2. Input Files

### 2.1. BPMN file (`.bpmn`)

- BPMN 2.0 XML file yang diexport dari BPMN modeller (Camunda Modeler, Bizagi, dll).
- Bisa berupa **collaboration diagram** (multiple pools, message flows) atau satu process saja.
- Contoh:

    diagram/case simple deadlock.bpmn

### 2.2. LTL properties file (`props.ltl`)

- File teks biasa.
- Setiap property didefinisikan sebagai:

    nama_property = LTL_FORMULA

- Baris yang diawali `#` dianggap komentar.
- Contoh `props.ltl` yang dipakai di project ini:

    # Loop deadlock: jika tetap berada di loop (G exLoop), sistem tetap harus bisa mencapai Final
    loop_deadlock = G ( F (G exLoop) -> G (F Final) )

    # Source deadlock: XOR-split yang tidak pernah mencapai XOR-join yang sesuai harus menuju deadlock
    source_deadlock = G ( ((exXORsplit) && ! F (exXORjoin)) -> F (deadlock) )

    # Improper structuring: XOR-split + pattern cabang + AND-join yang menunggu token selamanya
    improper_structuring_deadlock = G ( (((exXORsplit && path1) && !path2) && F (ANDjoin_waiting)) -> F (deadlock) )

**Penting:**  
Kalau ada baris non-komentar di `props.ltl` yang tidak mengandung `=`, program akan error:

    ValueError: Invalid property line (missing '=')

Jadi semua formula harus dalam format `nama = formula`.

---

## 3. Command Line Usage

Pola umum pemanggilan:

    python bpmn_ltl_checker.py --bpmn PATH_TO_BPMN --props PATH_TO_LTL_FILE --model-name "MODEL NAME" --out-json PATH_TO_JSON --out-cypher PATH_TO_CYPHER --out-graph-cypher PATH_TO_GRAPH_CYPHER --max-depth N

### 3.1. Argumen

- `--bpmn`  
  Path ke file BPMN 2.0 XML.  
  Contoh:

      --bpmn "./diagram/case simple deadlock.bpmn"

- `--props`  
  Path ke file properties LTL.  

      --props ./props.ltl

- `--model-name`  
  Label untuk run verifikasi ini (muncul di report dan bisa disimpan di Neo4j).  

      --model-name "case simple deadlock"

- `--out-json`  
  Path file JSON ringkasan hasil verifikasi.  

      --out-json "case-simple-deadlock.json"

- `--out-cypher`  
  Path file Cypher yang meng-*update* node di Neo4j dengan info pelanggaran (misal `violated = true`, alasan, dll).  

      --out-cypher "case-simple-deadlock.cql"

- `--out-graph-cypher`  
  Path file Cypher yang membangun **graph BPMN** di Neo4j (node = elemen BPMN, edge = sequence/message flow).  

      --out-graph-cypher "graph-case-simple-deadlock.cql"

- `--max-depth`  
  Batas maksimal kedalaman eksplorasi state space (finite-trace).  
  Berguna buat mencegah eksplorasi tak terbatas di model yang banyak loop.  

      --max-depth 50

---

## 4. Example: Running the Verifier

Dari root project (Windows PowerShell / CMD):

    python .\bpmn_ltl_checker.py --bpmn ".\diagram\case simple deadlock.bpmn" --props .\props.ltl --model-name "case simple deadlock" --out-json "case-simple-deadlock.json" --out-cypher "case-simple-deadlock.cql" --out-graph-cypher "graph-case-simple-deadlock.cql" --max-depth 50

Di Linux/macOS (satu baris):

    python bpmn_ltl_checker.py --bpmn "./diagram/case simple deadlock.bpmn" --props ./props.ltl --model-name "case simple deadlock" --out-json "case-simple-deadlock.json" --out-cypher "case-simple-deadlock.cql" --out-graph-cypher "graph-case-simple-deadlock.cql" --max-depth 50

Contoh output terminal:

    ╔═══════════════════════════════════════╗
    ║ case simple deadlock LTL VERIFICATION ║
    ╚═══════════════════════════════════════╝

    [1/5] Loading BPMN and building graph...
          ✓ Loaded 19 states, 22 transitions

    [2/5] Building Kripke structure...
          ✓ Using finite-trace semantics (LTLf)

    [3/5] Initializing LTL model checker...
          ✓ Ready

    [4/5] Loading LTL properties...
          ✓ 3 properties to verify

    [5/5] Verifying properties...

    [1/4] Checking: loop_deadlock
        Formula: G ( F (G exLoop) -> G (F Final) )
        ✓ PASSED

    [2/4] Checking: source_deadlock
        Formula: G ( ((exXORsplit) && ! F (exXORjoin)) -> F (deadlock) )
        ✗ FAILED
        Counterexample path: StartEvent_1gck240 -> Activity_0v59t9i -> ...

    ...

    ══════════════════════════════════════════════════════════════════════
    VERIFICATION REPORT
    ══════════════════════════════════════════════════════════════════════

    Total Properties: 3
      ✓ Passed: 2
      ✗ Failed: 1
      ⚠ Errors: 0

---

## 5. JSON Output

`--out-json` berisi ringkasan hasil dalam format terstruktur.

Contoh struktur (kurang lebih):

    {
      "model_name": "case simple deadlock",
      "bpmn_file": "diagram/case simple deadlock.bpmn",
      "num_states": 19,
      "num_transitions": 22,
      "properties": [
        {
          "name": "loop_deadlock",
          "formula": "G ( F (G exLoop) -> G (F Final) )",
          "status": "PASSED"
        },
        {
          "name": "source_deadlock",
          "formula": "G ( ((exXORsplit) && ! F (exXORjoin)) -> F (deadlock) )",
          "status": "FAILED",
          "counterexample": [
            "StartEvent_1gck240",
            "Activity_0v59t9i",
            "Gateway_1ht8p0l",
            "..."
          ]
        }
      ]
    }

JSON ini bisa dipakai buat laporan, analisis lanjut, atau integrasi ke UI lain.

---

## 6. Neo4j Visualisation

### 6.1. Import BPMN Graph

1. Jalankan Neo4j.
2. Buka Neo4j Browser.
3. (Opsional) Reset database:

       MATCH (n) DETACH DELETE n;

4. Jalankan file graph Cypher yang dihasilkan, contoh:

       graph-case-simple-deadlock.cql

   Caranya: buka file `.cql`, copy semua, paste di Neo4j Browser, klik **Run**.

   Setelah itu, elemen BPMN muncul sebagai node (misal label `BPMNNode` dengan property `id`, `name`, `type`, `modelName`, dll).

### 6.2. Import Violation Annotations

Setelah graph dasar ada, jalankan file Cypher pelanggaran:

    case-simple-deadlock.cql

Ini biasanya:

- Mencari node berdasarkan `id` BPMN  
- Menambahkan property seperti:
  - `n.violated = true`
  - `n.violations = ["source_deadlock"]`
  - `n.reason = "penjelasan singkat..."`

Contoh query di Neo4j:

    // Lihat semua node yang melanggar
    MATCH (n)
    WHERE n.violated = true
    RETURN n;

    // Hapus semua anotasi pelanggaran (sebelum ganti diagram)
    MATCH (n)
    REMOVE n.violated, n.violations, n.reason;

Untuk styling (warna/bentuk) di Browser, pakai `:style` dan atur misalnya node dengan `violated = true` berwarna merah.

---

## 7. Typical Workflow

1. **Model BPMN**  
   - Buat / edit BPMN collaboration diagram.  
   - Export sebagai `*.bpmn`.

2. **Definisikan LTL Properties**  
   - Edit `props.ltl`. Minimal tiga formula:
     - `loop_deadlock`
     - `source_deadlock`
     - `improper_structuring_deadlock`

3. **Jalankan Checker**  

       python bpmn_ltl_checker.py --bpmn ... --props ... --model-name ... --out-json ... --out-cypher ... --out-graph-cypher ... --max-depth 50

4. **Visualisasi (opsional)**  
   - Import `graph-*.cql` untuk bangun graph.  
   - Import `*.cql` (violations) untuk tandai node bermasalah.  

5. **Perbaiki Model & Ulangi**  
   - Perbaiki gateway / message flow yang menyebabkan deadlock.  
   - Run lagi sampai model sesuai.

---

## 8. Troubleshooting

### 8.1. Error `Invalid property line (missing '=')`

- Penyebab: ada baris di `props.ltl` yang bukan komentar tapi tidak mengandung `=`.
- Solusi: pastikan semua formula ditulis `nama = formula` dan komentar diawali `#`.

### 8.2. Program lama / seperti hang

- Penyebab: banyak loop dan `--max-depth` terlalu besar.
- Solusi:
  - Kurangi `--max-depth` (misalnya 30 atau 50).

### 8.3. Graph di Neo4j kosong

- Coba:

      MATCH (n) RETURN n LIMIT 10;

- Kalau tidak ada hasil:
  - Pastikan sudah menjalankan `graph-*.cql`.  
  - Cek apakah ada error saat menjalankan script.  
  - Baru setelah graph ada, jalankan file `*.cql` pelanggaran.

---