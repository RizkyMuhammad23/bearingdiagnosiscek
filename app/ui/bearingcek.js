// Port JavaScript dari src/predict.py + src/features.py.
// Model HistGradientBoosting diterima dari src/tampilan.py (diekspor dari artifacts/model.joblib).
(function (root) {
  const NUM_COLS = ["kecepatan_rpm", "beban_kN", "suhu_housing_C", "rms_g", "puncak_g", "kurtosis",
    "crest_factor", "energi_bpfo", "energi_bpfi", "energi_bsf", "frekuensi_dominan_hz"];
  const CLASSES = ["normal", "outer_race", "inner_race", "bola"];
  const WAJIB = ["posisi_sensor", "rms_g", "puncak_g", "energi_bpfo", "energi_bpfi", "energi_bsf", "frekuensi_dominan_hz"];
  const POSISI_VALID = ["drive_end", "fan_end"];
  const RENTANG_WAJAR = {
    kecepatan_rpm: [1000, 3000], beban_kN: [0.5, 6], suhu_housing_C: [35, 70], rms_g: [0.1, 1.2],
    puncak_g: [0.4, 4.5], kurtosis: [0, 7.5], crest_factor: [1.7, 5.2], energi_bpfo: [0, 0.7],
    energi_bpfi: [0, 0.5], energi_bsf: [0, 0.45], frekuensi_dominan_hz: [15, 290],
  };
  const BATAS_MUSTAHIL = {
    kecepatan_rpm: [0, 20000], beban_kN: [0, 500], suhu_housing_C: [-40, 200], rms_g: [0, 100],
    puncak_g: [0, 500], kurtosis: [-3, 200], crest_factor: [1, 100], energi_bpfo: [0, 100],
    energi_bpfi: [0, 100], energi_bsf: [0, 100], frekuensi_dominan_hz: [0, 20000],
  };
  const LABEL = {
    normal: "Normal", outer_race: "Cincin luar rusak (outer race)",
    inner_race: "Cincin dalam rusak (inner race)", bola: "Bola bearing rusak",
  };
  const SUKU_CADANG = {
    normal: "Tidak perlu penggantian. Lanjutkan pemantauan rutin.",
    outer_race: "Siapkan bearing pengganti (cincin luar rusak). Periksa juga dudukan housing dan kelurusan.",
    inner_race: "Siapkan bearing pengganti (cincin dalam rusak). Periksa kondisi poros dan fit poros-bearing.",
    bola: "Siapkan bearing pengganti (bola/elemen gelinding rusak). Periksa pelumasan dan kontaminasi grease.",
  };
  const EPS = 1e-4;
  const fmt = (x) => String(+x.toPrecision(6));

  class MasukanTidakValid extends Error {
    constructor(pesan) { super(pesan.join("; ")); this.pesan = pesan; }
  }

  const kosong = (v) => v === null || v === undefined || (typeof v === "string" && v.trim() === "") ||
    (typeof v === "number" && Number.isNaN(v));

  function validasi(m) {
    if (!m || typeof m !== "object" || Array.isArray(m)) {
      throw new MasukanTidakValid(["Masukan harus berupa objek berisi kolom-kolom rekaman."]);
    }
    const galat = [], peringatan = [], b = {};
    let pos = m.posisi_sensor;
    if (kosong(pos)) galat.push("posisi_sensor wajib diisi (drive_end atau fan_end).");
    else {
      pos = String(pos).trim().toLowerCase().replace(/ /g, "_").replace(/-/g, "_");
      if (!POSISI_VALID.includes(pos)) galat.push(`posisi_sensor '${m.posisi_sensor}' tidak dikenal. Pilih drive_end atau fan_end.`);
      b.posisi_sensor = pos;
    }
    for (const c of NUM_COLS) {
      const v = m[c];
      if (kosong(v)) {
        if (WAJIB.includes(c)) galat.push(`${c} wajib diisi.`);
        b[c] = NaN; continue;
      }
      const s = typeof v === "string" ? v.trim().replace(",", ".") : v;
      const x = typeof s === "string" ? (/^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(s) ? Number(s) : NaN) : Number(s);
      if (typeof s === "string" && /^[-+]?inf(inity)?$/i.test(s)) { galat.push(`${c} harus berupa angka terhingga.`); continue; }
      if (Number.isNaN(x)) { galat.push(`${c} harus berupa angka, bukan '${v}'.`); continue; }
      if (!Number.isFinite(x)) { galat.push(`${c} harus berupa angka terhingga.`); continue; }
      let y = x;
      if (c === "rms_g" && y < 0) {
        peringatan.push("rms_g bernilai negatif; dipakai nilai mutlaknya (besar getaran tidak mungkin negatif).");
        y = Math.abs(y);
      }
      if (c === "kecepatan_rpm" && y === 0) {
        peringatan.push("kecepatan_rpm = 0 dianggap sensor kecepatan gagal membaca; nilainya diisi median data latih.");
        b[c] = NaN; continue;
      }
      const [lo, hi] = BATAS_MUSTAHIL[c];
      if (!(y >= lo && y <= hi)) {
        galat.push(`${c} = ${fmt(y)} mustahil secara fisik (batas ${fmt(lo)} sampai ${fmt(hi)}). Periksa salah ketik atau sensor.`);
        continue;
      }
      const [wlo, whi] = RENTANG_WAJAR[c];
      if (!(y >= wlo && y <= whi)) {
        peringatan.push(`${c} = ${fmt(y)} di luar rentang data latih (${fmt(wlo)} sampai ${fmt(whi)}); prediksi kurang dapat dipercaya.`);
      }
      b[c] = y;
    }
    if (galat.length) throw new MasukanTidakValid(galat);

    const rms = b.rms_g, puncak = b.puncak_g;
    if (rms === 0) throw new MasukanTidakValid(["rms_g = 0 berarti tidak ada getaran sama sekali. Periksa sensor atau rekaman."]);
    if (puncak < rms) throw new MasukanTidakValid([`puncak_g (${fmt(puncak)}) tidak mungkin lebih kecil dari rms_g (${fmt(rms)}).`]);
    if (Number.isNaN(b.crest_factor)) {
      b.crest_factor = puncak / rms;
      peringatan.push("crest_factor kosong; dihitung otomatis dari puncak_g / rms_g.");
    } else if (Math.abs(b.crest_factor - puncak / rms) > 0.1 * (puncak / rms)) {
      peringatan.push(`crest_factor (${fmt(b.crest_factor)}) tidak cocok dengan puncak_g / rms_g (${(puncak / rms).toFixed(3)}). Periksa kembali rekaman.`);
    }
    for (const c of ["beban_kN", "suhu_housing_C", "kurtosis"]) {
      if (Number.isNaN(b[c])) peringatan.push(`${c} kosong; diisi median data latih.`);
    }
    if (Number.isNaN(b.kecepatan_rpm) && !peringatan.some((p) => p.includes("kecepatan_rpm = 0"))) {
      peringatan.push("kecepatan_rpm kosong; diisi median data latih. Fitur orde frekuensi tidak dapat dihitung.");
    }
    return [b, peringatan];
  }

  // src/features.py :: add_features
  function fitur(b) {
    const f = Object.assign({}, b);
    const rpm = b.kecepatan_rpm > 0 ? b.kecepatan_rpm : NaN;
    f.orde_frekuensi = b.frekuensi_dominan_hz / (rpm / 60);
    const eo = Math.max(b.energi_bpfo, 0), ei = Math.max(b.energi_bpfi, 0), eb = Math.max(b.energi_bsf, 0);
    const total = eo + ei + eb, denom = total + 3 * EPS;
    f.energi_total = total;
    f.porsi_bpfo = (eo + EPS) / denom;
    f.porsi_bpfi = (ei + EPS) / denom;
    f.porsi_bsf = (eb + EPS) / denom;
    f.log_rasio_bpfo_bpfi = Math.log((eo + EPS) / (ei + EPS));
    f.log_rasio_bsf_bpfi = Math.log((eb + EPS) / (ei + EPS));
    return f;
  }

  function vektor(M, b) {
    const f = fitur(b);
    const x = M.num_cols.map((c, i) => (Number.isNaN(f[c]) || f[c] === undefined ? M.medians[i] : f[c]));
    for (const k of M.cat) x.push(b.posisi_sensor === k ? 1 : 0);
    return x;
  }

  function nilaiPohon(t, x) {
    let i = 0;
    while (!t.leaf[i]) {
      const v = x[t.f[i]];
      i = Number.isNaN(v) ? (t.ml[i] ? t.l[i] : t.r[i]) : (v <= t.t[i] ? t.l[i] : t.r[i]);
    }
    return t.v[i];
  }

  function predictProba(M, x) {
    const raw = M.baseline.slice();
    for (const it of M.trees) for (let k = 0; k < it.length; k++) raw[k] += nilaiPohon(it[k], x);
    const mx = Math.max(...raw), e = raw.map((r) => Math.exp(r - mx)), s = e.reduce((a, c) => a + c, 0);
    const p = {};
    M.classes.forEach((c, k) => { p[c] = e[k] / s; });
    return p;
  }

  function prediksi(M, masukan) {
    const [b, peringatan] = validasi(masukan);
    const serius = peringatan.some((p) => p.includes("di luar rentang") || p.includes("kecepatan_rpm"));
    const ambang = M.meta.threshold.ambang_keyakinan;
    const p = predictProba(M, vektor(M, b));
    const prob = {};
    for (const c of CLASSES) prob[c] = Math.round(p[c] * 1e4) / 1e4;
    const kondisi = CLASSES.reduce((a, c) => (prob[c] > prob[a] ? c : a), CLASSES[0]);
    return {
      kondisi, kondisi_label: LABEL[kondisi], probabilitas: prob, keyakinan: prob[kondisi],
      ambang_keyakinan: ambang, perlu_cek_analis: prob[kondisi] < ambang || serius,
      suku_cadang: SUKU_CADANG[kondisi], peringatan, versi_model: M.meta.versi_model,
    };
  }

  const api = { NUM_COLS, CLASSES, WAJIB, RENTANG_WAJAR, LABEL, MasukanTidakValid, validasi, prediksi, predictProba, vektor };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.BearingModel = api;
})(typeof window !== "undefined" ? window : globalThis);
