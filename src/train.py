"""Pelatihan model diagnosis bearing, sama persis dengan notebooks/01_eda_sampai_model.ipynb.

Menghasilkan angka yang dilaporkan (macro F1 validation baseline 0,7247 dan final 0,8197;
macro F1 test baseline 0,7466 dan final 0,7991; ambang keyakinan 0,65), lalu menyimpan
artifacts/model.joblib dan artifacts/metadata.json.

Jalankan dari akar repositori:
    python -m src.train                 # simpan ke artifacts/
    python -m src.train --out tmp_art   # simpan ke folder lain
"""
from __future__ import annotations

import argparse
import datetime
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features import (
    CLASSES, ENGINEERED, NUM_COLS, RAW_FEATURES, RENTANG_WAJAR, SUKU_CADANG,
    clean_raw, feature_step, make_preprocessor,
)

ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42


def logreg_baseline():
    """Baseline wajib: regresi logistik multikelas dengan 12 fitur asli."""
    return Pipeline([
        ("prep", make_preprocessor(NUM_COLS, scale=True)),
        ("clf", LogisticRegression(max_iter=2000)),
    ])


def dengan_fitur(clf, scale=False):
    """Pipeline: fitur turunan (add_features) -> imputasi median (+ standardisasi) & one-hot -> model."""
    return Pipeline([
        ("fe", feature_step()),
        ("prep", make_preprocessor(NUM_COLS + ENGINEERED, scale=scale)),
        ("clf", clf),
    ])


def hgb(**kw):
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=800, min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=RANDOM_STATE, **kw)


KANDIDAT = {
    "Baseline: regresi logistik (fitur asli)": logreg_baseline(),
    "Regresi logistik + fitur turunan": dengan_fitur(LogisticRegression(max_iter=3000), scale=True),
    "Random forest + fitur turunan": dengan_fitur(RandomForestClassifier(
        n_estimators=500, min_samples_leaf=3, max_features=0.5, class_weight="balanced_subsample",
        n_jobs=-1, random_state=RANDOM_STATE)),
    "HistGradientBoosting (fitur asli)": Pipeline([
        ("prep", make_preprocessor(NUM_COLS, scale=False)),
        ("clf", hgb())]),
    "HistGradientBoosting + fitur turunan": dengan_fitur(hgb()),
    "HistGradientBoosting + fitur turunan (class_weight balanced)": dengan_fitur(hgb(class_weight="balanced")),
}
NAMA_BASELINE = "Baseline: regresi logistik (fitur asli)"


def main(out_dir: Path) -> None:
    # 1. Muat dan bersihkan data
    df_raw = pd.read_csv(ROOT / "data" / "raw.csv")
    df, log_bersih = clean_raw(df_raw)
    print(f"data: {len(df_raw)} baris mentah -> {len(df)} baris bersih")

    # 2. Bagi data stratified 70 / 15 / 15
    X, y = df[RAW_FEATURES], df["kondisi"]
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp,
                                                    random_state=RANDOM_STATE)
    print(f"train {len(X_train)} | validation {len(X_val)} | test {len(X_test)}")

    # 3. Latih semua kandidat di train, nilai dengan macro F1 di validation
    hasil = {}
    for nama, model in KANDIDAT.items():
        model.fit(X_train, y_train)
        hasil[nama] = f1_score(y_val, model.predict(X_val), average="macro")
    print("\nmacro F1 validation:")
    for nama, skor in sorted(hasil.items(), key=lambda kv: -kv[1]):
        print(f"  {skor:.4f}  {nama}")
    nama_final = max(hasil, key=hasil.get)
    print("model terpilih:", nama_final)

    # 4. Ambang keyakinan: nilai terkecil yang membuat tebakan lolos ambang benar >= 90% (validation)
    model_val = KANDIDAT[nama_final]
    proba = model_val.predict_proba(X_val)
    tebak = model_val.classes_[proba.argmax(1)]
    yakin = proba.max(1)
    ambang = None
    for t in np.round(np.arange(0.40, 0.96, 0.05), 2):
        lolos = yakin >= t
        if (tebak[lolos] == y_val.values[lolos]).mean() >= 0.90:
            ambang = float(t)
            break
    print("ambang keyakinan:", ambang)

    # 5. Latih ulang final dan baseline dengan train + validation, lalu buka data test SEKALI
    X_trval, y_trval = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
    model_final = clone(KANDIDAT[nama_final]).fit(X_trval, y_trval)
    model_base = clone(KANDIDAT[NAMA_BASELINE]).fit(X_trval, y_trval)
    pred_test = model_final.predict(X_test)
    f1_final_test = f1_score(y_test, pred_test, average="macro")
    f1_base_test = f1_score(y_test, model_base.predict(X_test), average="macro")
    print(f"\nmacro F1 test: baseline {f1_base_test:.4f} | final {f1_final_test:.4f}")
    print(classification_report(y_test, pred_test, labels=CLASSES, digits=3))
    print(pd.DataFrame(confusion_matrix(y_test, pred_test, labels=CLASSES), index=CLASSES, columns=CLASSES))

    # 6. Simpan model dan metadata
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_final, out_dir / "model.joblib", compress=3)
    rep = classification_report(y_test, pred_test, labels=CLASSES, output_dict=True)
    metadata = {
        "nama_model": "bearing-fault-classifier",
        "versi_model": "1.0.0",
        "dibuat": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "kasus": "04 Diagnosis Kerusakan Bearing",
        "algoritma": nama_final,
        "parameter": {k: v for k, v in model_final.named_steps["clf"].get_params().items()
                      if k in ["learning_rate", "max_iter", "min_samples_leaf", "l2_regularization", "class_weight"]},
        "jumlah_iterasi_boosting": int(model_final.named_steps["clf"].n_iter_),
        "dilatih_dengan": "train + validation",
        "fitur_masukan": {
            "posisi_sensor": {"tipe": "kategori", "nilai": ["drive_end", "fan_end"]},
            **{c: {"tipe": "angka", "rentang_wajar": list(RENTANG_WAJAR[c])} for c in NUM_COLS},
        },
        "fitur_turunan": ENGINEERED,
        "kelas": CLASSES,
        "aturan_keputusan": "argmax probabilitas (kelas dengan probabilitas tertinggi)",
        "threshold": {
            "ambang_keyakinan": ambang,
            "arti": "jika probabilitas tertinggi < ambang, prediksi tetap ditampilkan tetapi ditandai perlu dicek analis",
            "dipilih_dari": "data validation: ambang terkecil dengan akurasi >= 0.90 pada prediksi yang lolos",
        },
        "metrik": {
            "metrik_utama": "macro F1",
            "baseline": {"nama": NAMA_BASELINE, "macro_f1_validation": round(float(hasil[NAMA_BASELINE]), 4),
                         "macro_f1_test": round(float(f1_base_test), 4)},
            "final": {"macro_f1_validation": round(float(hasil[nama_final]), 4),
                      "macro_f1_test": round(float(f1_final_test), 4),
                      "akurasi_test": round(float(accuracy_score(y_test, pred_test)), 4),
                      "f1_per_kelas_test": {k: round(rep[k]["f1-score"], 4) for k in CLASSES}},
        },
        "data": {"baris_mentah": int(len(df_raw)), "baris_bersih": int(len(df)),
                 "train": int(len(X_train)), "validation": int(len(X_val)), "test": int(len(X_test)),
                 "pembersihan": log_bersih, "random_state": RANDOM_STATE},
        "suku_cadang": SUKU_CADANG,
        "lingkungan": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                       "pandas": pd.__version__, "numpy": np.__version__},
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\ntersimpan:", out_dir / "model.joblib", "dan", out_dir / "metadata.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Latih model diagnosis bearing")
    ap.add_argument("--out", default=str(ROOT / "artifacts"), help="folder keluaran (default: artifacts/)")
    main(Path(ap.parse_args().out))
