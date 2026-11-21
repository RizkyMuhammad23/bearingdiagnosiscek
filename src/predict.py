"""Logika prediksi bersama untuk API (FastAPI) dan aplikasi (Streamlit).

Keduanya memanggil fungsi yang sama di sini, sehingga validasi masukan dan hasil prediksi selalu konsisten.
"""
from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import (  # noqa: E402
    CLASSES,
    NUM_COLS,
    POSISI_VALID,
    RAW_FEATURES,
    RENTANG_WAJAR,
    SUKU_CADANG,
)

ARTIFACTS = ROOT / "artifacts"

# Kolom yang wajib diisi: tanpa ini tidak ada informasi getaran yang bisa dibaca.
WAJIB = ["posisi_sensor", "rms_g", "puncak_g", "energi_bpfo", "energi_bpfi", "energi_bsf", "frekuensi_dominan_hz"]
# Kolom yang boleh kosong: model mengisinya dengan median data latih (sama seperti saat pelatihan).
BOLEH_KOSONG = ["kecepatan_rpm", "beban_kN", "suhu_housing_C", "kurtosis", "crest_factor"]

# Batas fisik: di luar ini masukan pasti salah ketik / sensor rusak, jadi ditolak.
BATAS_MUSTAHIL = {
    "kecepatan_rpm": (0.0, 20000.0),
    "beban_kN": (0.0, 500.0),
    "suhu_housing_C": (-40.0, 200.0),
    "rms_g": (0.0, 100.0),
    "puncak_g": (0.0, 500.0),
    "kurtosis": (-3.0, 200.0),
    "crest_factor": (1.0, 100.0),
    "energi_bpfo": (0.0, 100.0),
    "energi_bpfi": (0.0, 100.0),
    "energi_bsf": (0.0, 100.0),
    "frekuensi_dominan_hz": (0.0, 20000.0),
}

LABEL_INDONESIA = {
    "normal": "Normal",
    "outer_race": "Cincin luar rusak (outer race)",
    "inner_race": "Cincin dalam rusak (inner race)",
    "bola": "Bola bearing rusak",
}


class MasukanTidakValid(ValueError):
    """Masukan ditolak. Atribut `pesan` berisi daftar alasan yang bisa ditampilkan ke pengguna."""

    def __init__(self, pesan: list[str]):
        super().__init__("; ".join(pesan))
        self.pesan = pesan


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(ARTIFACTS / "model.joblib")


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    return json.loads((ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))


def _kosong(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip() == "") or (isinstance(v, float) and math.isnan(v))


def validasi(masukan: dict) -> tuple[dict, list[str]]:
    """Periksa satu rekaman. Mengembalikan (baris_siap_prediksi, peringatan).

    Melempar MasukanTidakValid jika masukan tidak bisa diprediksi sama sekali.
    Peringatan tidak menghentikan prediksi, tetapi ditampilkan ke teknisi.
    """
    if not isinstance(masukan, dict):
        raise MasukanTidakValid(["Masukan harus berupa objek JSON / kamus berisi kolom-kolom rekaman."])

    galat: list[str] = []
    peringatan: list[str] = []
    baris: dict[str, Any] = {}

    # Posisi sensor
    posisi = masukan.get("posisi_sensor")
    if _kosong(posisi):
        galat.append("posisi_sensor wajib diisi (drive_end atau fan_end).")
    else:
        posisi = str(posisi).strip().lower().replace(" ", "_").replace("-", "_")
        if posisi not in POSISI_VALID:
            galat.append(f"posisi_sensor '{masukan.get('posisi_sensor')}' tidak dikenal. Pilih drive_end atau fan_end.")
        baris["posisi_sensor"] = posisi

    # Kolom angka
    for col in NUM_COLS:
        v = masukan.get(col)
        if _kosong(v):
            if col in WAJIB:
                galat.append(f"{col} wajib diisi.")
            baris[col] = float("nan")
            continue
        try:
            x = float(str(v).replace(",", ".")) if isinstance(v, str) else float(v)
        except (TypeError, ValueError):
            galat.append(f"{col} harus berupa angka, bukan '{v}'.")
            continue
        if math.isinf(x) or math.isnan(x):
            galat.append(f"{col} harus berupa angka terhingga.")
            continue
        if col == "rms_g" and x < 0:
            peringatan.append("rms_g bernilai negatif; dipakai nilai mutlaknya (besar getaran tidak mungkin negatif).")
            x = abs(x)
        if col == "kecepatan_rpm" and x == 0:
            peringatan.append("kecepatan_rpm = 0 dianggap sensor kecepatan gagal membaca; nilainya diisi median data latih.")
            baris[col] = float("nan")
            continue
        lo, hi = BATAS_MUSTAHIL[col]
        if not (lo <= x <= hi):
            galat.append(f"{col} = {x:g} mustahil secara fisik (batas {lo:g} sampai {hi:g}). Periksa salah ketik atau sensor.")
            continue
        wlo, whi = RENTANG_WAJAR[col]
        if not (wlo <= x <= whi):
            peringatan.append(
                f"{col} = {x:g} di luar rentang data latih ({wlo:g} sampai {whi:g}); prediksi kurang dapat dipercaya."
            )
        baris[col] = x

    if galat:
        raise MasukanTidakValid(galat)

    # Pemeriksaan antar kolom
    rms, puncak = baris["rms_g"], baris["puncak_g"]
    if rms == 0:
        raise MasukanTidakValid(["rms_g = 0 berarti tidak ada getaran sama sekali. Periksa sensor atau rekaman."])
    if puncak < rms:
        raise MasukanTidakValid([f"puncak_g ({puncak:g}) tidak mungkin lebih kecil dari rms_g ({rms:g})."])
    cf = baris.get("crest_factor")
    if cf is None or math.isnan(cf):
        baris["crest_factor"] = puncak / rms
        peringatan.append("crest_factor kosong; dihitung otomatis dari puncak_g / rms_g.")
    elif abs(cf - puncak / rms) > 0.1 * (puncak / rms):
        peringatan.append(
            f"crest_factor ({cf:g}) tidak cocok dengan puncak_g / rms_g ({puncak / rms:.3f}). Periksa kembali rekaman."
        )

    for col in ["beban_kN", "suhu_housing_C", "kurtosis"]:
        if math.isnan(baris[col]):
            peringatan.append(f"{col} kosong; diisi median data latih.")
    if math.isnan(baris["kecepatan_rpm"]) and not any("kecepatan_rpm = 0" in p for p in peringatan):
        peringatan.append("kecepatan_rpm kosong; diisi median data latih. Fitur orde frekuensi tidak dapat dihitung.")

    return baris, peringatan


def prediksi(masukan: dict) -> dict:
    """Prediksi satu rekaman. Mengembalikan kamus siap dikirim sebagai JSON."""
    baris, peringatan = validasi(masukan)
    # Masukan di luar rentang data latih atau tanpa kecepatan putar membuat prediksi kurang dapat dipercaya,
    # jadi hasilnya selalu ditandai untuk dicek analis walaupun probabilitasnya tinggi.
    serius = any(("di luar rentang" in p) or ("kecepatan_rpm" in p) for p in peringatan)
    model = load_model()
    meta = load_metadata()
    ambang = float(meta["threshold"]["ambang_keyakinan"])

    X = pd.DataFrame([baris])[RAW_FEATURES]
    proba = model.predict_proba(X)[0]
    prob = {k: float(p) for k, p in zip(model.classes_, proba)}
    prob = {k: round(prob[k], 4) for k in CLASSES}
    kondisi = max(prob, key=prob.get)
    yakin = prob[kondisi] >= ambang

    return {
        "kondisi": kondisi,
        "kondisi_label": LABEL_INDONESIA[kondisi],
        "probabilitas": prob,
        "keyakinan": prob[kondisi],
        "ambang_keyakinan": ambang,
        "perlu_cek_analis": (not yakin) or serius,
        "suku_cadang": SUKU_CADANG[kondisi],
        "peringatan": peringatan,
        "versi_model": meta["versi_model"],
    }


def prediksi_banyak(df: pd.DataFrame) -> pd.DataFrame:
    """Prediksi banyak rekaman (misalnya dari CSV). Baris yang ditolak tidak menghentikan proses."""
    hasil = []
    for _, r in df.iterrows():
        d = {k: (None if pd.isna(v) else v) for k, v in r.items()}
        try:
            out = prediksi(d)
            hasil.append({
                "status": "ok",
                "kondisi": out["kondisi"],
                "keyakinan": out["keyakinan"],
                "perlu_cek_analis": out["perlu_cek_analis"],
                **{f"p_{k}": v for k, v in out["probabilitas"].items()},
                "catatan": " | ".join(out["peringatan"]),
            })
        except MasukanTidakValid as e:
            hasil.append({"status": "ditolak", "catatan": " | ".join(e.pesan)})
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(hasil)], axis=1)
