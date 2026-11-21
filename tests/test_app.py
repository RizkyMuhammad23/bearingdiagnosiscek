"""Tes cepat: jalankan dengan `python -m pytest -q` dari akar repositori."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import MasukanTidakValid, prediksi, prediksi_banyak  # noqa: E402
from src.features import CLASSES, clean_raw  # noqa: E402

CONTOH = json.loads((ROOT / "data" / "contoh_masukan.json").read_text())


@pytest.mark.parametrize("kelas", CLASSES)
def test_contoh_per_kelas(kelas):
    hasil = prediksi(CONTOH[kelas])
    assert hasil["kondisi"] == kelas
    assert abs(sum(hasil["probabilitas"].values()) - 1) < 1e-3


@pytest.mark.parametrize("masukan", [{}, {"posisi_sensor": "atas"}, "bukan dict",
                                     {**CONTOH["normal"], "rms_g": "abc"},
                                     {**CONTOH["normal"], "puncak_g": 0.01},
                                     {**CONTOH["normal"], "energi_bsf": -1},
                                     {**CONTOH["normal"], "frekuensi_dominan_hz": float("inf")}])
def test_masukan_mustahil_ditolak(masukan):
    with pytest.raises(MasukanTidakValid):
        prediksi(masukan)


def test_masukan_aneh_diproses_dengan_peringatan():
    hasil = prediksi(CONTOH["aneh_tapi_diproses"])
    assert hasil["peringatan"]
    assert hasil["perlu_cek_analis"]


def test_clean_raw_konsisten():
    df, log = clean_raw(pd.read_csv(ROOT / "data" / "raw.csv"))
    assert set(df["kondisi"]) == set(CLASSES)
    assert log["duplikat_dihapus"] == 220
    assert (df["rms_g"] >= 0).all()


def test_prediksi_banyak_tidak_berhenti():
    df = pd.read_csv(ROOT / "data" / "raw.csv").head(50)
    df["rms_g"] = df["rms_g"].astype(object)
    df.loc[0, "rms_g"] = "abc"
    df.loc[1, "posisi_sensor"] = None
    out = prediksi_banyak(df.drop(columns=["kondisi"]))
    assert len(out) == 50
    assert (out["status"] == "ditolak").sum() == 2
