"""Aplikasi BearingCek untuk teknisi pemeliharaan (Streamlit).

Jalankan dari akar repositori:
    streamlit run app/ui/streamlit_app.py

Saat menyala, aplikasi memuat artifacts/model.joblib dan metadata.json, lalu menampilkan antarmuka
BearingCek (app/ui/web). Validasi masukan dan prediksi di antarmuka itu mengikuti aturan yang sama
dengan app/core/predict.py, yang juga dipakai API FastAPI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.export_web import build_page  # noqa: E402

st.set_page_config(page_title="BearingCek · Diagnosis Bearing", page_icon="⚙️", layout="wide",
                   initial_sidebar_state="collapsed")

# Buang kerangka bawaan Streamlit supaya halaman BearingCek mengisi layar penuh.
st.markdown(
    """
    <style>
      header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"],
      [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
      .stApp { background: #e6e9e7; }
      .block-container, [data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
      iframe { display: block; border: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Memuat model…")
def halaman() -> str:
    return build_page()


components.html(halaman(), height=1480, scrolling=True)
