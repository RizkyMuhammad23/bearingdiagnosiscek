"""Pembersihan data dan pembuatan fitur untuk diagnosis kerusakan bearing.

Modul ini dipakai di dua tempat:
1. notebook pelatihan (notebooks/01_eda_sampai_model.ipynb), dan
2. saat prediksi di API maupun aplikasi Streamlit.

Dengan begitu, fitur yang dilihat model saat dilatih sama persis dengan fitur
yang dihitung saat model dipakai.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Urutan kelas yang dipakai di seluruh proyek (laporan, confusion matrix, aplikasi).
CLASSES = ["normal", "outer_race", "inner_race", "bola"]

# Semua variasi penulisan label yang ditemukan di raw.csv, dipetakan ke 4 kelas baku.
LABEL_MAP = {
    "normal": "normal",
    "ok": "normal",
    "outer_race": "outer_race",
    "outer race": "outer_race",
    "outer": "outer_race",
    "or": "outer_race",
    "inner_race": "inner_race",
    "inner race": "inner_race",
    "inner": "inner_race",
    "ir": "inner_race",
    "bola": "bola",
    "ball": "bola",
}

# Kolom masukan asli dari sistem klien (tanpa id_rekaman dan target).
NUM_COLS = [
    "kecepatan_rpm",
    "beban_kN",
    "suhu_housing_C",
    "rms_g",
    "puncak_g",
    "kurtosis",
    "crest_factor",
    "energi_bpfo",
    "energi_bpfi",
    "energi_bsf",
    "frekuensi_dominan_hz",
]
CAT_COLS = ["posisi_sensor"]
RAW_FEATURES = CAT_COLS + NUM_COLS
POSISI_VALID = ["drive_end", "fan_end"]

# Fitur tambahan hasil rekayasa (dibuat oleh add_features).
ENGINEERED = [
    "orde_frekuensi",
    "energi_total",
    "porsi_bpfo",
    "porsi_bpfi",
    "porsi_bsf",
    "log_rasio_bpfo_bpfi",
    "log_rasio_bsf_bpfi",
]
MODEL_FEATURES = RAW_FEATURES + ENGINEERED

# Saran suku cadang per kelas, ditampilkan di aplikasi untuk teknisi.
SUKU_CADANG = {
    "normal": "Tidak perlu penggantian. Lanjutkan pemantauan rutin.",
    "outer_race": "Siapkan bearing pengganti (cincin luar rusak). Periksa juga dudukan housing dan kelurusan.",
    "inner_race": "Siapkan bearing pengganti (cincin dalam rusak). Periksa kondisi poros dan fit poros-bearing.",
    "bola": "Siapkan bearing pengganti (bola/elemen gelinding rusak). Periksa pelumasan dan kontaminasi grease.",
}

# Rentang wajar dari data latih (persentil 0,1 dan 99,9 dibulatkan ke luar).
# Dipakai aplikasi untuk memberi peringatan, bukan untuk menolak masukan.
RENTANG_WAJAR = {
    "kecepatan_rpm": (1000.0, 3000.0),
    "beban_kN": (0.5, 6.0),
    "suhu_housing_C": (35.0, 70.0),
    "rms_g": (0.1, 1.2),
    "puncak_g": (0.4, 4.5),
    "kurtosis": (0.0, 7.5),
    "crest_factor": (1.7, 5.2),
    "energi_bpfo": (0.0, 0.7),
    "energi_bpfi": (0.0, 0.5),
    "energi_bsf": (0.0, 0.45),
    "frekuensi_dominan_hz": (15.0, 290.0),
}

_EPS = 1e-4


def normalize_label(series: pd.Series) -> pd.Series:
    """Ubah label yang ditulis bermacam-macam (NORMAL, OR, Ball, ...) menjadi 4 kelas baku."""
    return series.astype(str).str.strip().str.lower().map(LABEL_MAP)


def clean_raw(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Bersihkan data mentah. Mengembalikan (data_bersih, catatan_jumlah_baris_per_masalah).

    Langkah:
    - hapus baris duplikat persis (rekaman yang terekspor dua kali);
    - seragamkan label target;
    - rms_g negatif diubah menjadi nilai mutlaknya, karena puncak_g / |rms_g| tetap
      cocok dengan crest_factor (artinya hanya tandanya yang salah, besarnya benar);
    - kecepatan_rpm = 0 dianggap hilang (NaN) karena motor yang diam tidak mungkin
      menghasilkan getaran dengan frekuensi dominan puluhan hertz.
    Nilai hilang tidak diisi di sini; pengisian dilakukan oleh imputer di dalam
    pipeline model supaya statistiknya hanya dihitung dari data latih.
    """
    log = {}
    out = df.copy()

    n0 = len(out)
    out = out.drop_duplicates()
    log["duplikat_dihapus"] = n0 - len(out)

    if "kondisi" in out.columns:
        raw_label = out["kondisi"].astype(str).str.strip()
        log["label_tidak_baku"] = int((~raw_label.isin(CLASSES)).sum())
        out["kondisi"] = normalize_label(out["kondisi"])
        log["label_tidak_dikenal_dibuang"] = int(out["kondisi"].isna().sum())
        out = out.dropna(subset=["kondisi"])

    neg = out["rms_g"] < 0
    log["rms_negatif_diperbaiki"] = int(neg.sum())
    out.loc[neg, "rms_g"] = out.loc[neg, "rms_g"].abs()

    zero_rpm = out["kecepatan_rpm"] <= 0
    log["rpm_nol_dijadikan_kosong"] = int(zero_rpm.sum())
    out.loc[zero_rpm, "kecepatan_rpm"] = np.nan

    log["suhu_kosong"] = int(out["suhu_housing_C"].isna().sum())
    log["kurtosis_kosong"] = int(out["kurtosis"].isna().sum())
    return out.reset_index(drop=True), log


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan fitur turunan. Aman dipanggil pada data latih maupun satu baris masukan aplikasi.

    - orde_frekuensi: frekuensi dominan dibagi frekuensi putar poros (rpm/60).
      BPFO, BPFI, dan BSF sebanding dengan kecepatan putar, jadi kelipatan putaran
      lebih stabil daripada frekuensi mentah ketika rpm berubah-ubah.
    - porsi_*: bagian energi tiap frekuensi khas terhadap total energi ketiganya.
      Analis getaran membaca spektrum dengan cara ini: frekuensi mana yang menonjol.
    - log_rasio_*: perbandingan energi antar frekuensi khas dalam skala log.
    """
    X = X.copy()
    for c in NUM_COLS:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    rpm = X["kecepatan_rpm"].where(X["kecepatan_rpm"] > 0)
    X["orde_frekuensi"] = X["frekuensi_dominan_hz"] / (rpm / 60.0)
    e_o = X["energi_bpfo"].clip(lower=0)
    e_i = X["energi_bpfi"].clip(lower=0)
    e_b = X["energi_bsf"].clip(lower=0)
    total = e_o + e_i + e_b
    X["energi_total"] = total
    denom = total + 3 * _EPS
    X["porsi_bpfo"] = (e_o + _EPS) / denom
    X["porsi_bpfi"] = (e_i + _EPS) / denom
    X["porsi_bsf"] = (e_b + _EPS) / denom
    X["log_rasio_bpfo_bpfi"] = np.log((e_o + _EPS) / (e_i + _EPS))
    X["log_rasio_bsf_bpfi"] = np.log((e_b + _EPS) / (e_i + _EPS))
    return X


def make_preprocessor(numeric_cols: list[str], scale: bool = True):
    """ColumnTransformer: imputasi median (+ standardisasi) untuk angka, one-hot untuk posisi sensor."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), numeric_cols),
            (
                "cat",
                OneHotEncoder(categories=[POSISI_VALID], handle_unknown="ignore"),
                CAT_COLS,
            ),
        ]
    )


def feature_step():
    """Langkah pipeline yang menjalankan add_features."""
    from sklearn.preprocessing import FunctionTransformer

    return FunctionTransformer(add_features, validate=False)
