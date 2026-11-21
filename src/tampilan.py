"""Komponen tampilan: bangun halaman BearingCek (app/ui/bearingcek.html) dari model di artifacts/.

Model HistGradientBoosting di dalam pipeline dibaca dari model.joblib, lalu pohon-pohonnya,
median imputer, dan urutan fiturnya diekspor ke JSON. Halaman memakai JSON itu untuk menghitung
prediksi di browser dengan aturan yang sama persis seperti src/predict.py
(lihat app/ui/bearingcek.js; kesamaan hasil diuji di tests/test_web_parity.py).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from src.predict import ROOT, load_metadata, load_model

UI = ROOT / "app" / "ui"


def export_model(model=None, meta=None) -> dict:
    model = model or load_model()
    meta = meta or load_metadata()
    clf = model.named_steps["clf"]
    prep = model.named_steps["prep"]
    imputer = prep.named_transformers_["num"].named_steps["imputer"]
    num_cols = [n.split("__", 1)[1] for n in prep.get_feature_names_out() if n.startswith("num__")]

    trees = []
    for iterasi in clf._predictors:
        per_kelas = []
        for pohon in iterasi:
            n = pohon.nodes
            if n["is_categorical"].any():
                raise ValueError("Pohon dengan split kategorikal belum didukung oleh bearingcek.js")
            per_kelas.append({
                "f": n["feature_idx"].tolist(),
                "t": [float(x) for x in n["num_threshold"]],
                "l": n["left"].tolist(),
                "r": n["right"].tolist(),
                "leaf": n["is_leaf"].astype(int).tolist(),
                "v": [float(x) for x in n["value"]],
                "ml": n["missing_go_to_left"].astype(int).tolist(),
            })
        trees.append(per_kelas)

    return {
        "classes": clf.classes_.tolist(),
        "baseline": clf._baseline_prediction.ravel().tolist(),
        "num_cols": num_cols,
        "medians": imputer.statistics_.tolist(),
        "cat": ["drive_end", "fan_end"],
        "trees": trees,
        "meta": meta,
    }


def build_page() -> str:
    """Kembalikan HTML lengkap halaman BearingCek, siap ditampilkan."""
    html = (UI / "bearingcek.html").read_text(encoding="utf-8")
    contoh = (ROOT / "data" / "contoh_masukan.json").read_text(encoding="utf-8")
    model_js = (UI / "bearingcek.js").read_text(encoding="utf-8")
    cm = ROOT / "figures" / "fig06_confusion_matrix_test.png"
    cm_uri = "data:image/png;base64," + base64.b64encode(cm.read_bytes()).decode() if cm.exists() else ""
    model_json = json.dumps(export_model(), separators=(",", ":")).replace("</", "<\\/")
    return (html.replace("__MODELJS__", model_js)
                .replace("__MODEL_JSON__", model_json)
                .replace("__CONTOH__", contoh)
                .replace("__CM__", cm_uri))


if __name__ == "__main__":
    out = Path("bearingcek.html")
    out.write_text(build_page(), encoding="utf-8")
    print("ditulis:", out.resolve())
