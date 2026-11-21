# Diagnosis Kerusakan Bearing · Tim Bearing

**Kasus:** 04 Diagnosis Kerusakan *Bearing* · **Aplikasi:** https://bearing-diagnosis-mkb.streamlit.app · **Video:** [tautan video] · **Penjelasan kode:** [buka di Google Colab](https://colab.research.google.com/github/RizkyMuhammad23/bearing-diagnosis/blob/main/notebooks/penjelasan_kode_colab.ipynb)

| Anggota | NIM | Bagian yang dikerjakan |
|---|---|---|
| Rizky Muhammad Ikhsan | 2403329 | EDA, pembersihan data, fitur, model, dan evaluasi |
| Muhammad Khairul Irham | 2402023 | API dan aplikasi, deploy, laporan, video |

## Masalah dan pengguna

Analis getaran di pabrik semen harus membaca ribuan rekaman getaran *bearing* sendirian. Model membaca ringkasan
satu rekaman lalu menebak kondisinya: `normal`, `outer_race` (cincin luar rusak), `inner_race` (cincin dalam rusak),
atau `bola`. Penggunanya teknisi pemeliharaan, yang dari hasil itu tahu suku cadang apa yang perlu disiapkan dan
rekaman mana yang tetap harus dilihat analis.

## Data dan pembersihan

| Masalah yang ditemukan | Jumlah baris | Cara menangani |
|---|---|---|
| Baris duplikat persis | 220 | `drop_duplicates` sebelum data dibagi |
| Label `kondisi` ditulis dalam 16 variasi (`NORMAL`, `ok`, `OR`, `BALL`, ...) | 915 | dipetakan ke 4 kelas baku |
| `kurtosis` kosong | 456 | diisi median data *train* di dalam *pipeline* |
| `suhu_housing_C` kosong | 153 | diisi median data *train* di dalam *pipeline* |
| `kecepatan_rpm` = 0 (sensor gagal membaca) | 61 | jadi kosong, lalu diisi median |
| `rms_g` negatif (hanya tandanya salah) | 45 | diambil nilai mutlaknya |

Rincian dan alasannya ada di `data/log_pembersihan.csv`. Setelah dibersihkan tersisa 15.000 rekaman
(`data/bersih.csv`): `normal` 54,8%, `outer_race` 20,3%, `inner_race` 14,5%, `bola` 10,4%.

## Hasil (macro F1)

| Model | macro F1 *validation* | macro F1 *test* |
|---|---|---|
| *Baseline*: regresi logistik, fitur asli | 0,7247 | 0,7466 |
| *Random forest* + fitur turunan | 0,8057 | tidak dibuka |
| HistGradientBoosting + fitur turunan (final) | 0,8197 | **0,7991** |

F1 per kelas di *test*: `normal` 0,888, `outer_race` 0,834, `inner_race` 0,780, `bola` 0,694. Fitur terpenting
(`artifacts/importance.csv`) adalah `orde_frekuensi`, yaitu frekuensi dominan dibagi frekuensi putar poros, disusul
suhu, kecepatan, dan beban. Kelas dipilih dengan *argmax*; jika probabilitas tertinggi di bawah **ambang keyakinan
0,65**, hasilnya ditandai "perlu dicek analis". Di *test*, 84,8% rekaman lolos ambang dengan akurasi 0,905.

## Cara menjalankan

```bash
pip install -r requirements.txt
python -m src.train                                   # latih ulang model dan artifacts (sekitar 30 detik)
uvicorn app.api.main:app --reload                     # API (POST /predict, dokumentasi di /docs)
streamlit run app/ui/streamlit_app.py                 # aplikasi BearingCek
python -m pytest -q                                   # tes
```

Contoh API:
```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"posisi_sensor":"drive_end","kecepatan_rpm":2525,"beban_kN":2.71,"suhu_housing_C":55.5,"rms_g":0.6077,
       "puncak_g":1.998,"kurtosis":3.216,"crest_factor":3.288,"energi_bpfo":0.2455,"energi_bpfi":0.1137,
       "energi_bsf":0.0341,"frekuensi_dominan_hz":41.2}'
```

Untuk mencoba unggah banyak rekaman di aplikasi, pakai `data/contoh_upload.csv` (40 rekaman, termasuk masukan
tidak wajar dan mustahil).

## Struktur repositori

```
notebooks/analisis.ipynb              EDA sampai evaluasi dan penyimpanan model
notebooks/penjelasan_kode_colab.ipynb penjelasan kode untuk video (dibuka di Google Colab)
data/                                 raw.csv, bersih.csv, log_pembersihan.csv, contoh masukan untuk aplikasi
src/features.py                       pembersihan dan fitur (dipakai saat latih dan prediksi)
src/train.py, predict.py              pelatihan, serta validasi masukan dan prediksi
src/tampilan.py                       komponen tampilan (bangun halaman BearingCek dari model)
artifacts/                            model.joblib, metadata.json, importance.csv, hasil_validation.csv
app/                                  API FastAPI dan antarmuka Streamlit (BearingCek)
figures/                              grafik dan tangkapan layar untuk laporan
docs/                                 laporan (PDF/Word), skrip video, brief kasus
tests/                                tes validasi masukan dan kesamaan hasil aplikasi dengan predict.py
```

Aplikasi Streamlit memuat `artifacts/model.joblib` saat menyala, lalu `src/tampilan.py` mengirim pohon-pohon model ke
halaman BearingCek (`app/ui/bearingcek.html`), sehingga diagnosis dihitung di browser dengan aturan yang sama seperti
`src/predict.py`. API memakai `src/predict.py` secara langsung.

## Deploy

Aplikasi berjalan di Streamlit Community Cloud: repositori ini, *branch* `main`, *main file*
`app/ui/streamlit_app.py`, Python 3.13. Setiap *push* ke `main` otomatis memperbarui aplikasi. Aplikasi gratis
tertidur jika lama tidak dibuka, jadi buka beberapa menit sebelum demo.

API bersifat opsional dan tidak dipanggil oleh aplikasi. Untuk menaruhnya di Render: *New → Web Service*, *build*
`pip install -r requirements.txt`, *start* `uvicorn app.api.main:app --host 0.0.0.0 --port $PORT`, dan variabel
lingkungan `PYTHON_VERSION=3.13.0`.

## Penanganan masukan tidak wajar

Masukan mustahil ditolak dengan alasannya (API membalas kode 422): kolom wajib kosong, teks di kolom angka, nilai
tak hingga, energi negatif, `puncak_g` < `rms_g`, `rms_g` = 0, atau posisi sensor tidak dikenal. Masukan aneh tetap
diprediksi dengan catatan: `rms_g` negatif diambil nilai mutlaknya, `kecepatan_rpm` = 0 dan kolom opsional kosong
diisi median, `crest_factor` kosong dihitung dari puncak/RMS, dan nilai di luar rentang data latih diberi
peringatan serta ditandai "perlu dicek analis".

## Keterbatasan

- Kesalahan terbesar adalah *bearing* rusak yang ditebak normal (232 dari 2.250 rekaman *test*), terutama jika
  frekuensi dominannya di 1× atau 2× putaran. Kelas `bola` paling lemah (F1 0,694).
- Data sintetis dan hanya ringkasan: satu jenis *bearing*, kecepatan 1.200 sampai 3.000 rpm.
- Kerusakan ganda tidak ada di data, jadi model selalu memilih satu kelas.
- Tidak ada identitas motor atau waktu, sehingga tren dari rekaman sebelumnya belum bisa dipakai.

## Daftar periksa sebelum dikumpulkan

- [ ] Repositori publik dan semua anggota punya *commit*
- [x] Notebook bisa dijalankan dari awal sampai akhir
- [x] Baseline dilaporkan dan dibandingkan
- [x] Data test dibuka satu kali, di akhir
- [x] `artifacts/metadata.json` berisi versi, fitur, parameter, *threshold*, dan metrik
- [x] Aplikasi dideploy dan bisa dibuka dari ponsel
- [x] Masukan tidak wajar ditolak atau ditandai dengan pesan jelas
- [ ] Laporan PDF maksimal 6 halaman, video maksimal 5 menit
