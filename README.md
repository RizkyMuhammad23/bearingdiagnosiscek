# Diagnosis Kerusakan Bearing · Tim Bearing

**Kasus:** 04 Diagnosis Kerusakan *Bearing* · **Aplikasi:** [tautan Streamlit] · **API:** [tautan API, opsional] · **Video:** [tautan video]

| Anggota | NIM | Bagian yang dikerjakan |
|---|---|---|
| Rizky Muhammad Ikhsan | 2403329 | EDA, pembersihan data, fitur (`src/features.py`), model dan evaluasi (notebook, `artifacts/`) |
| Muhammad Khairul Irham | 2402023 | API dan aplikasi (`app/`), deploy, laporan, video |

## Masalah dan pengguna

Analis getaran di sebuah pabrik semen harus membaca ribuan rekaman getaran *bearing* sendirian. Model ini
membaca ringkasan satu rekaman (RMS, puncak, kurtosis, energi di frekuensi khas kerusakan, frekuensi dominan,
kecepatan, beban, suhu) lalu menebak kondisinya: `normal`, `outer_race` (cincin luar rusak), `inner_race`
(cincin dalam rusak), atau `bola` (bola rusak). Penggunanya teknisi pemeliharaan, yang dari hasil itu tahu
suku cadang apa yang perlu disiapkan dan rekaman mana yang tetap harus dilihat analis.

## Data dan pembersihan

Data mentah berisi 15.220 rekaman. Setelah dibersihkan tersisa 15.000 rekaman dengan empat kelas: `normal`
54,8%, `outer_race` 20,3%, `inner_race` 14,5%, dan `bola` 10,4%.

| Masalah yang ditemukan | Jumlah baris | Cara menangani |
|---|---|---|
| Baris duplikat persis (`id_rekaman` juga sama) | 220 | Dihapus dengan `drop_duplicates` sebelum data dibagi, supaya rekaman yang sama tidak masuk *train* dan *test* sekaligus |
| Label `kondisi` ditulis dalam 16 variasi (`NORMAL`, `ok`, `OR`, `Outer Race`, `BALL`, `IR`, ...) | 915 | Dipetakan ke 4 kelas baku (`LABEL_MAP` di `src/features.py`) |
| `kurtosis` kosong | 456 | Diisi median data *train* oleh imputer di dalam *pipeline* |
| `suhu_housing_C` kosong | 153 | Sama seperti di atas. Kosongnya merata di semua kelas (sekitar 1% dan 3%), jadi tidak membawa informasi |
| `kecepatan_rpm` = 0 | 61 | Dijadikan kosong lalu diisi median. Motor yang diam tidak mungkin punya frekuensi dominan puluhan hertz, jadi ini sensor yang gagal membaca |
| `rms_g` negatif | 45 | Diambil nilai mutlaknya. `puncak_g / abs(rms_g)` tetap sama dengan `crest_factor` (selisih maksimum 0,0008), jadi hanya tandanya yang salah |

Jumlah baris dihitung dari data mentah (sebelum duplikat dihapus).

## Hasil

Data dibagi acak secara *stratified*: *train* 10.500, *validation* 2.250, *test* 2.250 rekaman.

| Model | *Macro* F1 *validation* | *Macro* F1 *test* |
|---|---|---|
| *Baseline*: regresi logistik multikelas, fitur asli | 0,725 | 0,747 |
| HistGradientBoosting + fitur turunan (final) | 0,820 | 0,799 |

F1 per kelas di data *test*: `normal` 0,888, `outer_race` 0,834, `inner_race` 0,780, `bola` 0,694.

Keputusan kelas memakai *argmax* (kelas dengan probabilitas tertinggi). *Threshold* yang dipakai adalah
**ambang keyakinan 0,65**: jika probabilitas tertinggi di bawah 0,65, aplikasi tetap menampilkan tebakan
tetapi menandainya "perlu dicek analis". Ambang ini dipilih dari data *validation* sebagai nilai terkecil yang
membuat tebakan yang lolos ambang benar minimal 90%. Alasannya bisnis: kesalahan terbanyak model adalah
*bearing* rusak yang ditebak normal, dan *bearing* rusak yang lolos pemeriksaan tetap dipakai sampai gagal.
Di data *test*, 84,8% rekaman lolos ambang dengan akurasi 0,905. Sisanya (15,2%) akurasinya hanya 0,516,
jadi memang perlu dilihat manusia.

Fitur yang paling berpengaruh (*permutation importance* di data *validation*) adalah `orde_frekuensi`, yaitu
frekuensi dominan dibagi frekuensi putar poros. Kerusakan cincin luar muncul di sekitar 3,6× putaran, cincin
dalam di sekitar 5,4×, dan bola di sekitar 2,3×, sedangkan rekaman normal di 1× dan 2×.

## Cara menjalankan

```bash
pip install -r requirements-dev.txt   # cukup requirements.txt jika hanya menjalankan aplikasi

# latih ulang dari awal (EDA sampai artifacts/), sekitar 1 menit
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_sampai_model.ipynb

uvicorn app.api.main:app --reload       # API, dokumentasi di http://127.0.0.1:8000/docs
streamlit run app/ui/streamlit_app.py   # aplikasi
python -m pytest -q                     # tes
```

Contoh memanggil API:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "content-type: application/json" -d '{
  "posisi_sensor": "drive_end", "kecepatan_rpm": 2525, "beban_kN": 2.71, "suhu_housing_C": 55.5,
  "rms_g": 0.6077, "puncak_g": 1.998, "kurtosis": 3.216, "crest_factor": 3.288,
  "energi_bpfo": 0.2455, "energi_bpfi": 0.1137, "energi_bsf": 0.0341, "frekuensi_dominan_hz": 41.2}'
```

Aplikasi Streamlit menampilkan antarmuka BearingCek (`app/ui/web/`). Saat menyala, aplikasi memuat
`artifacts/model.joblib`, mengekspor pohon-pohon model ke halaman (`app/core/export_web.py`), lalu halaman
menghitung prediksi di browser dengan aturan yang sama seperti `app/core/predict.py`. Kesamaan hasil keduanya
diuji di `tests/test_web_parity.py` (butuh Node.js). Panduan deploy ada di [DEPLOY.md](DEPLOY.md).

## Struktur repositori

```
notebooks/      notebook dari EDA sampai penyimpanan model
data/           data mentah (raw.csv); data bersih dibuat ulang oleh clean_raw()
src/features.py pembersihan dan pembuatan fitur, dipakai saat pelatihan dan saat prediksi
artifacts/      model.joblib + metadata.json
app/core/       validasi masukan dan prediksi, dipakai bersama oleh API dan aplikasi
app/api/        API (FastAPI)
app/ui/         antarmuka: streamlit_app.py + halaman BearingCek di app/ui/web/
reports/        grafik, tangkapan layar aplikasi, laporan
tests/          tes validasi masukan dan prediksi
docs/           brief kasus dan naskah video
```

## Penanganan masukan tidak wajar

Masukan yang mustahil ditolak dengan pesan yang menjelaskan alasannya: kolom wajib kosong, teks di kolom angka,
nilai tak hingga, energi negatif, `puncak_g` lebih kecil dari `rms_g`, `rms_g` = 0, atau posisi sensor yang
tidak dikenal. API mengembalikan kode 422 beserta daftar alasannya.

Masukan yang aneh tetapi masih bisa diproses tetap diprediksi dengan catatan: `rms_g` negatif diambil nilai
mutlaknya, `kecepatan_rpm` = 0 atau kolom opsional yang kosong diisi median, `crest_factor` kosong dihitung
dari puncak dibagi RMS, dan nilai di luar rentang data latih diberi peringatan. Jika ada nilai di luar rentang
atau kecepatan putarnya tidak diketahui, hasilnya otomatis ditandai "perlu dicek analis".

## Keterbatasan

- Kesalahan terbesar adalah *bearing* rusak yang ditebak normal (232 dari 2.250 rekaman *test*), terutama
  ketika frekuensi dominannya masih di 1× atau 2× putaran. Kelas `bola` paling lemah (F1 0,69).
- Data bersifat sintetis dan hanya berisi ringkasan. Model belum pernah melihat spektrum lengkap, jenis
  *bearing* lain, atau kecepatan di luar 1.200 sampai 3.000 rpm.
- Kerusakan ganda (misalnya cincin luar dan bola sekaligus) tidak ada di data, jadi model selalu memilih satu.
- Tidak ada informasi motor atau waktu, sehingga model tidak bisa memakai tren dari rekaman sebelumnya.

## Daftar periksa sebelum dikumpulkan

- [ ] Repositori bersifat publik dan setiap anggota memiliki *commit*
- [x] *Notebook* dapat dijalankan ulang dari awal sampai akhir tanpa galat
- [x] *Baseline* wajib dilaporkan dan dibandingkan
- [x] Data *test* hanya dibuka satu kali, di akhir
- [x] `artifacts/metadata.json` berisi versi model, fitur masukan, *threshold*, dan metrik
- [ ] Aplikasi dapat dibuka orang lain melalui tautan, termasuk dari ponsel
- [x] Aplikasi tidak berhenti mendadak ketika menerima masukan yang tidak wajar
- [ ] Laporan paling banyak 6 halaman dan video paling lama 5 menit
