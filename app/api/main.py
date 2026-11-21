"""REST API diagnosis bearing.

Jalankan dari akar repositori:
    uvicorn app.api.main:app --reload

Dokumentasi interaktif tersedia di http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional, Union

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # agar src/ bisa diimpor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.predict import CLASSES, MasukanTidakValid, load_metadata, load_model, prediksi

@asynccontextmanager
async def lifespan(_app):
    # Muat model sekali saat server menyala, bukan di setiap permintaan.
    load_model()
    load_metadata()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="API Diagnosis Kerusakan Bearing",
    description="Kasus 04 · Machine Learning MKB UPI. Menerima ringkasan satu rekaman getaran dan "
    "mengembalikan kondisi bearing, probabilitas tiap kelas, dan suku cadang yang perlu disiapkan.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

Angka = Optional[Union[float, str]]


class Rekaman(BaseModel):
    """Satu rekaman getaran. Kolom angka boleh dikirim sebagai angka atau teks; validasinya dilakukan di server."""

    posisi_sensor: Optional[str] = Field(None, examples=["drive_end"])
    kecepatan_rpm: Angka = Field(None, examples=[1794])
    beban_kN: Angka = Field(None, examples=[3.53])
    suhu_housing_C: Angka = Field(None, examples=[50.1])
    rms_g: Angka = Field(None, examples=[0.4344])
    puncak_g: Angka = Field(None, examples=[1.2863])
    kurtosis: Angka = Field(None, examples=[1.878])
    crest_factor: Angka = Field(None, examples=[2.961])
    energi_bpfo: Angka = Field(None, examples=[0.0729])
    energi_bpfi: Angka = Field(None, examples=[0.0015])
    energi_bsf: Angka = Field(None, examples=[0.1221])
    frekuensi_dominan_hz: Angka = Field(None, examples=[58.9])


class Hasil(BaseModel):
    kondisi: str
    kondisi_label: str
    probabilitas: dict[str, float]
    keyakinan: float
    ambang_keyakinan: float
    perlu_cek_analis: bool
    suku_cadang: str
    peringatan: list[str]
    versi_model: str


@app.get("/")
def root():
    return {"pesan": "API diagnosis bearing aktif. Buka /docs untuk mencoba.", "kelas": CLASSES}


@app.get("/health")
def health():
    meta = load_metadata()
    return {"status": "ok", "versi_model": meta["versi_model"]}


@app.get("/metadata")
def metadata():
    return load_metadata()


@app.post("/predict", response_model=Hasil)
def predict(rekaman: Rekaman):
    try:
        return prediksi(rekaman.model_dump())
    except MasukanTidakValid as e:
        raise HTTPException(status_code=422, detail={"pesan": "Masukan tidak valid", "alasan": e.pesan})
