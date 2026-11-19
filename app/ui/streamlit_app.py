"""Aplikasi Streamlit untuk teknisi pemeliharaan.

Jalankan dari akar repositori:
    streamlit run app/ui/streamlit_app.py

Secara bawaan aplikasi memanggil model langsung (cocok untuk Streamlit Community Cloud).
Jika variabel lingkungan atau secret API_URL diisi (misalnya https://nama-api.onrender.com),
aplikasi mengirim masukan ke API FastAPI tersebut.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.predict import (  # noqa: E402
    CLASSES,
    LABEL_INDONESIA,
    MasukanTidakValid,
    load_metadata,
    prediksi,
    prediksi_banyak,
)
from src.features import NUM_COLS, RAW_FEATURES, RENTANG_WAJAR  # noqa: E402

st.set_page_config(page_title="Diagnosis Bearing", page_icon="⚙️", layout="wide")

META = load_metadata()
CONTOH = json.loads((ROOT / "app" / "core" / "contoh_masukan.json").read_text())


def _api_url() -> str | None:
    url = os.environ.get("API_URL")
    if not url:
        try:
            url = st.secrets.get("API_URL")
        except Exception:
            url = None
    return url.rstrip("/") if url else None


API_URL = _api_url()

KETERANGAN = {
    "kecepatan_rpm": ("Kecepatan putar (rpm)", 1.0, "%.0f"),
    "beban_kN": ("Beban (kN)", 0.1, "%.2f"),
    "suhu_housing_C": ("Suhu housing (°C)", 0.1, "%.1f"),
    "rms_g": ("RMS getaran (g)", 0.01, "%.4f"),
    "puncak_g": ("Puncak getaran (g)", 0.01, "%.4f"),
    "kurtosis": ("Kurtosis", 0.1, "%.3f"),
    "crest_factor": ("Crest factor", 0.1, "%.3f"),
    "energi_bpfo": ("Energi BPFO (cincin luar)", 0.01, "%.4f"),
    "energi_bpfi": ("Energi BPFI (cincin dalam)", 0.01, "%.4f"),
    "energi_bsf": ("Energi BSF (bola)", 0.01, "%.4f"),
    "frekuensi_dominan_hz": ("Frekuensi dominan (Hz)", 0.1, "%.1f"),
}
WAJIB = {"rms_g", "puncak_g", "energi_bpfo", "energi_bpfi", "energi_bsf", "frekuensi_dominan_hz"}
TOMBOL_CONTOH = {
    "Normal": "normal",
    "Cincin luar": "outer_race",
    "Cincin dalam": "inner_race",
    "Bola": "bola",
    "Kasus ragu": "ragu",
    "Tidak wajar": "aneh_tapi_diproses",
    "Mustahil": "mustahil_ditolak",
}


def isi_contoh(nama: str | None) -> None:
    data = CONTOH.get(nama, {}) if nama else {}
    st.session_state["posisi_sensor"] = data.get("posisi_sensor") or "(kosong)"
    for c in NUM_COLS:
        st.session_state[c] = data.get(c)


def panggil_model(masukan: dict) -> dict:
    """Kembalikan hasil prediksi, atau {'ditolak': [alasan...]} jika masukan tidak valid."""
    if API_URL:
        import requests

        r = requests.post(f"{API_URL}/predict", json=masukan, timeout=30)
        if r.status_code == 422:
            detail = r.json().get("detail")
            if isinstance(detail, dict):
                return {"ditolak": detail.get("alasan", [str(detail)])}
            return {"ditolak": [str(detail)]}
        r.raise_for_status()
        return r.json()
    try:
        return prediksi(masukan)
    except MasukanTidakValid as e:
        return {"ditolak": e.pesan}


if "posisi_sensor" not in st.session_state:
    isi_contoh("normal")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Tentang model")
    m = META["metrik"]
    st.metric("Macro F1 (data test)", f"{m['final']['macro_f1_test']:.3f}",
              delta=f"{m['final']['macro_f1_test'] - m['baseline']['macro_f1_test']:+.3f} vs baseline")
    st.caption(f"Baseline regresi logistik: {m['baseline']['macro_f1_test']:.3f}")
    st.write(f"Model: {META['algoritma']}  \nVersi: {META['versi_model']}  \n"
             f"Ambang keyakinan: {META['threshold']['ambang_keyakinan']}")
    st.caption("Mode: memanggil API di " + API_URL if API_URL else "Mode: model dimuat langsung di aplikasi")

st.title("⚙️ Diagnosis Kerusakan Bearing")
st.write("Masukkan ringkasan satu rekaman getaran. Aplikasi menebak kondisi bearing, menampilkan probabilitas "
         "setiap kelas, dan menyarankan suku cadang yang perlu disiapkan.")

tab1, tab2, tab3 = st.tabs(["Satu rekaman", "Banyak rekaman (CSV)", "Cara kerja model"])

# ---------------------------------------------------------------- tab 1
with tab1:
    st.write("Isi contoh cepat:")
    tombol = list(TOMBOL_CONTOH.items()) + [("Kosongkan", None)]
    for awal in range(0, len(tombol), 4):
        for kol, (label, kunci) in zip(st.columns(4), tombol[awal:awal + 4]):
            kol.button(label, on_click=isi_contoh, args=(kunci,), use_container_width=True)

    with st.form("form_rekaman"):
        st.selectbox("Posisi sensor *", ["(kosong)", "drive_end", "fan_end"], key="posisi_sensor",
                     help="drive_end = sisi beban, fan_end = sisi kipas")
        cols = st.columns(3)
        for i, c in enumerate(NUM_COLS):
            label, step, fmt = KETERANGAN[c]
            lo, hi = RENTANG_WAJAR[c]
            cols[i % 3].number_input(label + (" *" if c in WAJIB else ""), key=c, value=None, step=step,
                                     format=fmt, help=f"Rentang data latih: {lo:g} sampai {hi:g}")
        st.caption("Tanda \\* berarti wajib diisi. Kolom lain boleh kosong; model mengisinya dengan median data latih.")
        kirim = st.form_submit_button("Diagnosis", type="primary", use_container_width=True)

    if kirim:
        masukan = {"posisi_sensor": None if st.session_state["posisi_sensor"] == "(kosong)"
                   else st.session_state["posisi_sensor"]}
        masukan.update({c: st.session_state.get(c) for c in NUM_COLS})
        try:
            hasil = panggil_model(masukan)
        except Exception as e:  # jaringan ke API putus, dsb.
            st.error(f"Gagal menghubungi API: {e}")
            hasil = None

        if hasil and "ditolak" in hasil:
            st.error("Masukan ditolak, prediksi tidak dibuat:")
            for alasan in hasil["ditolak"]:
                st.markdown(f"- {alasan}")
        elif hasil:
            label = hasil["kondisi_label"]
            if hasil["perlu_cek_analis"]:
                st.warning(f"### {label}\nPrediksi ini kurang meyakinkan atau masukannya tidak wajar. "
                           "Mintalah analis getaran memeriksa spektrum rekaman ini sebelum mengambil tindakan.")
            elif hasil["kondisi"] == "normal":
                st.success(f"### {label}")
            else:
                st.error(f"### {label}")

            a, b = st.columns([1, 2])
            with a:
                st.metric("Keyakinan model", f"{hasil['keyakinan']:.0%}",
                          help=f"Di bawah {hasil['ambang_keyakinan']:.0%} ditandai perlu dicek analis.")
                st.markdown(f"**Suku cadang:** {hasil['suku_cadang']}")
            with b:
                prob = pd.DataFrame({"kelas": [LABEL_INDONESIA[k] for k in CLASSES],
                                     "probabilitas": [hasil["probabilitas"][k] for k in CLASSES]})
                st.bar_chart(prob, x="kelas", y="probabilitas", horizontal=True, height=220)
            if hasil["peringatan"]:
                st.info("Catatan tentang masukan:\n" + "\n".join(f"- {p}" for p in hasil["peringatan"]))

# ---------------------------------------------------------------- tab 2
with tab2:
    st.write("Unggah CSV berisi banyak rekaman dengan kolom yang sama seperti data klien. "
             "Kolom `id_rekaman` dan `kondisi` boleh ada dan akan dibiarkan apa adanya.")
    st.code(",".join(RAW_FEATURES), language=None)
    berkas = st.file_uploader("Pilih berkas CSV", type=["csv"])
    if berkas is not None:
        try:
            df_in = pd.read_csv(berkas)
        except Exception as e:
            st.error(f"Berkas tidak dapat dibaca sebagai CSV: {e}")
            df_in = None
        if df_in is not None:
            kurang = [c for c in RAW_FEATURES if c not in df_in.columns]
            if kurang:
                st.error("Kolom berikut tidak ada di berkas: " + ", ".join(kurang))
            elif len(df_in) == 0:
                st.error("Berkas tidak berisi baris data.")
            elif len(df_in) > 20000:
                st.error("Maksimal 20.000 baris per unggahan.")
            else:
                with st.spinner(f"Memproses {len(df_in)} rekaman..."):
                    hasil_df = prediksi_banyak(df_in)
                ok = hasil_df["status"].eq("ok")
                c1, c2, c3 = st.columns(3)
                c1.metric("Diproses", int(ok.sum()))
                c2.metric("Ditolak", int((~ok).sum()))
                c3.metric("Perlu dicek analis", int(hasil_df.loc[ok, "perlu_cek_analis"].fillna(False).astype(bool).sum()))
                if ok.any():
                    st.bar_chart(hasil_df.loc[ok, "kondisi"].value_counts().reindex(CLASSES, fill_value=0))
                st.dataframe(hasil_df, use_container_width=True, height=360)
                buf = io.StringIO()
                hasil_df.to_csv(buf, index=False)
                st.download_button("Unduh hasil (CSV)", buf.getvalue(), "hasil_diagnosis.csv", "text/csv")

# ---------------------------------------------------------------- tab 3
with tab3:
    st.markdown(
        """
Model ini dilatih dengan 15.000 rekaman getaran berlabel dari sistem klien setelah duplikat dihapus dan
label diseragamkan. Alurnya: data mentah → pembersihan → fitur turunan (`src/features.py`) →
HistGradientBoosting → API FastAPI / aplikasi ini.

Fitur terpenting adalah **orde frekuensi**, yaitu frekuensi dominan dibagi frekuensi putar poros. Kerusakan
cincin luar muncul di sekitar 3,6× putaran, cincin dalam di sekitar 5,4×, dan bola di sekitar 2,3×.

Keterbatasan: kesalahan terbanyak adalah bearing rusak yang ditebak normal, terutama ketika frekuensi
dominannya masih di 1× atau 2× putaran. Hasil yang ditandai *perlu dicek analis* sebaiknya tidak langsung
dipakai.
"""
    )
    m = META["metrik"]
    st.dataframe(pd.DataFrame({
        "Model": ["Baseline: regresi logistik", META["algoritma"]],
        "Macro F1 validation": [m["baseline"]["macro_f1_validation"], m["final"]["macro_f1_validation"]],
        "Macro F1 test": [m["baseline"]["macro_f1_test"], m["final"]["macro_f1_test"]],
    }), hide_index=True)
    gambar = ROOT / "reports" / "figures" / "fig06_confusion_matrix_test.png"
    if gambar.exists():
        st.image(str(gambar), caption="Confusion matrix pada data test")
