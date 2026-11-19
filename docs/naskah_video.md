# Naskah video demo (maksimal 5 menit)

Tim berdua: Anggota 1 = Rizky (data, fitur, model), Anggota 2 = [nama] (API, aplikasi, deploy).
Setiap anggota menjelaskan bagian yang dikerjakannya sendiri. Angka di bawah sama dengan `artifacts/metadata.json`.

Sebelum merekam: buka aplikasi Streamlit, buka `/docs` API jika API dipasang (bangunkan dulu kalau memakai
Render gratis), dan siapkan berkas CSV kecil untuk tab "Banyak rekaman".

## 0:00–0:30 · Masalah dan pengguna · Anggota 2

- Klien: analis getaran di pabrik semen. Hanya dia yang bisa membaca spektrum, rekamannya ribuan.
- Pengguna aplikasi: teknisi pemeliharaan. Keluaran: kondisi *bearing*, probabilitas tiap kelas, suku cadang.
- Empat kelas: normal, cincin luar, cincin dalam, bola. Metrik: *macro* F1.

## 0:30–2:00 · Arsitektur data → model → API → aplikasi · Anggota 1 (data, fitur, model), lalu Anggota 2 (API, aplikasi)

- Tampilkan Gambar 5 laporan (`reports/figures/fig07_arsitektur.png`).
- Data: 15.220 rekaman, enam masalah data (tunjukkan tabel di README). Contoh paling menarik: `rms_g` negatif
  ternyata hanya salah tanda, karena puncak dibagi nilai mutlaknya tetap sama dengan *crest factor*.
- Fitur: `orde_frekuensi` = frekuensi dominan dibagi rpm/60. Tunjukkan Gambar 2: tiap kerusakan punya
  kelipatan putaran khas (3,6× cincin luar, 5,4× cincin dalam, sekitar 2,3× bola).
- `src/features.py` dipakai saat pelatihan dan saat prediksi, jadi fiturnya selalu sama.
- Model disimpan sebagai satu *pipeline* di `artifacts/model.joblib` plus `metadata.json`.
- `app/core/predict.py` dipakai bersama oleh API FastAPI dan Streamlit.

## 2:00–3:30 · Demo langsung · Anggota 2

1. Klik contoh "Cincin luar" → Diagnosis. Tunjukkan kelas, keyakinan, grafik probabilitas, suku cadang.
2. Klik "Kasus ragu" → Diagnosis. Keyakinan di bawah 65%, jadi ditandai perlu dicek analis.
3. Masukan tidak wajar: klik "Tidak wajar" (rpm 0, RMS negatif, suhu kosong, beban 9 kN). Aplikasi tetap
   memprediksi, menampilkan catatan, dan menandai perlu dicek analis.
4. Klik "Mustahil" atau "Kosongkan" → Diagnosis. Masukan ditolak dengan alasan, aplikasi tidak berhenti.
5. (Jika waktu cukup) unggah CSV di tab "Banyak rekaman" dan tunjukkan baris yang ditolak tidak menghentikan proses.

## 3:30–4:30 · Hasil, *baseline*, keterbatasan · Anggota 1

- *Macro* F1 *test*: *baseline* regresi logistik 0,747, model final HistGradientBoosting 0,799.
- Tunjukkan *confusion matrix*: kesalahan terbanyak adalah rusak ditebak normal (232 dari 2.250), antar jenis
  kerusakan jarang tertukar. Kelas bola paling lemah (F1 0,69).
- Ambang keyakinan 0,65: 84,8% rekaman lolos dengan akurasi 0,905; sisanya diserahkan ke analis.
- Keterbatasan: data sintetis, satu jenis *bearing*, rpm 1.200 sampai 3.000, tidak ada kerusakan ganda.

## 4:30–5:00 · Cadangan

Jangan diisi. Sisakan agar video tetap di bawah 5 menit.
