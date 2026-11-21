"""Pastikan antarmuka BearingCek (app/ui/bearingcek.js) memberi hasil yang sama dengan src/predict.py.

Butuh Node.js; tes dilewati jika `node` tidak tersedia.
"""
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tampilan import build_page, export_model  # noqa: E402
from src.predict import MasukanTidakValid, prediksi  # noqa: E402

NODE = shutil.which("node")
HARNESS = """
const fs = require('fs'); const B = require(process.argv[2]);
const M = JSON.parse(fs.readFileSync(process.argv[3])); const cases = JSON.parse(fs.readFileSync(process.argv[4]));
const out = cases.map((c) => { try { const h = B.prediksi(M, c); return {ok: true, kondisi: h.kondisi, prob: h.probabilitas, cek: h.perlu_cek_analis, warn: h.peringatan}; }
  catch (e) { return {ok: false, err: e.pesan || [String(e.message)]}; } });
process.stdout.write(JSON.stringify(out));
"""


def test_halaman_terbangun():
    html = build_page()
    for token in ["__MODELJS__", "__MODEL_JSON__", "__CONTOH__", "__CM__"]:
        assert token not in html


@pytest.mark.skipif(NODE is None, reason="node tidak tersedia")
def test_js_sama_dengan_python(tmp_path):
    df = pd.read_csv(ROOT / "data" / "raw.csv").drop(columns=["id_rekaman", "kondisi"])
    aneh = df[(df.kecepatan_rpm == 0) | (df.rms_g < 0) | df["kurtosis"].isna()]
    contoh = json.loads((ROOT / "data" / "contoh_masukan.json").read_text())
    cases = [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
             for r in pd.concat([df.sample(500, random_state=0), aneh.head(200)]).to_dict("records")]
    cases += list(contoh.values()) + [{}, {**contoh["normal"], "rms_g": "abc"}, {**contoh["normal"], "crest_factor": None}]

    (tmp_path / "m.json").write_text(json.dumps(export_model()))
    (tmp_path / "c.json").write_text(json.dumps(cases))
    (tmp_path / "h.js").write_text(HARNESS)
    js = json.loads(subprocess.run([NODE, str(tmp_path / "h.js"), str(ROOT / "app" / "ui" / "bearingcek.js"),
                                    str(tmp_path / "m.json"), str(tmp_path / "c.json")],
                                   capture_output=True, text=True, check=True).stdout)
    for c, j in zip(cases, js):
        try:
            p = prediksi(c)
        except MasukanTidakValid as e:
            assert not j["ok"] and j["err"] == e.pesan
            continue
        assert j["ok"] and j["kondisi"] == p["kondisi"] and j["cek"] == p["perlu_cek_analis"] and j["warn"] == p["peringatan"]
        for k, v in p["probabilitas"].items():
            assert abs(j["prob"][k] - v) < 1e-4
