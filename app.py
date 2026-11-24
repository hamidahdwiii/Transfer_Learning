"""
⚡ GridCast Panama — Dashboard perbandingan 5 model prediksi beban listrik.
Jalankan lokal:  streamlit run app.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "panama_hourly_load.csv")
METRICS_PATH = os.path.join(ROOT, "results", "metrics.csv")
FORECASTS_PATH = os.path.join(ROOT, "results", "forecasts.csv")

st.set_page_config(page_title="GridCast Panama | Benchmark Model Prediksi",
                   page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# =====================================================================
# KATALOG MODEL  (warna tetap per model — tidak berubah saat difilter)
# =====================================================================
MODELS = {
    "Chronos-2": dict(color="#2a78d6", family="Foundation Model", maker="Amazon", icon="🧠",
                      short="Transformer pra-latih dari Amazon. Membaca pola dari ribuan deret waktu lain, "
                            "lalu langsung memprediksi data Panama tanpa dilatih ulang (zero-shot)."),
    "TimesFM": dict(color="#eb6834", family="Foundation Model", maker="Google Research", icon="🔭",
                    short="TimesFM 2.5 (200 juta parameter). Model ala GPT untuk deret waktu: memotong data "
                          "menjadi potongan (patch) lalu menebak potongan berikutnya."),
    "Moirai": dict(color="#1baf7a", family="Foundation Model", maker="Salesforce", icon="🌐",
                   short="Moirai 2.0. Dilatih pada data dari berbagai frekuensi (menit sampai tahunan) "
                         "dan menghasilkan rentang ketidakpastian lewat kuantil."),
    "Lag-Llama": dict(color="#eda100", family="Foundation Model", maker="Rasul dkk., 2024 · open-source", icon="🦙",
                      short="Arsitektur mirip LLaMA yang memakai nilai-nilai masa lalu (lag: 1 jam, 1 hari, "
                            "1 minggu lalu, dst.) sebagai bahan prediksi. Menghasilkan ratusan skenario."),
    "SARIMAX": dict(color="#6250d6", family="Statistik", maker="Statistik klasik", icon="📐",
                    short="Seasonal ARIMA + variabel eksogen (X): pola harian 24 jam, pola mingguan, "
                          "hari libur, dan masa sekolah. Dilatih ulang pada 6 minggu terakhir tiap jendela."),
    "Seasonal Naive": dict(color="#8a8f98", family="Baseline", maker="Patokan", icon="📏",
                           short="Tebakan paling sederhana: beban jam ini = beban di jam yang sama seminggu lalu. "
                                 "Model yang tidak bisa mengalahkannya tidak layak dipakai."),
}
REQUESTED = ["Chronos-2", "TimesFM", "Moirai", "Lag-Llama", "SARIMAX"]
HORIZON_LABEL = {24: "24 jam · 1 hari", 168: "168 jam · 1 minggu", 720: "720 jam · 1 bulan"}
ACTUAL_COLOR = "#1f2430"

FONT = "Plus Jakarta Sans, Inter, system-ui, sans-serif"

# =====================================================================
# GAYA
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input { font-family: 'Plus Jakarta Sans', Inter, sans-serif; }
.block-container { padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1350px; }
[data-testid="stSidebar"] { background: #f6f7fb; border-right: 1px solid #e6e8f0; }
h1, h2, h3 { letter-spacing: -0.02em; }

/* HERO */
.hero { position: relative; overflow: hidden; border-radius: 22px; padding: 34px 38px 30px;
  background: radial-gradient(1200px 400px at 85% -20%, rgba(98,80,214,.55), transparent 60%),
              radial-gradient(800px 300px at 0% 120%, rgba(42,120,214,.45), transparent 60%),
              linear-gradient(135deg, #0c1230 0%, #141a44 55%, #1b1450 100%);
  color: #eef0ff; margin-bottom: 18px; }
.hero:after { content: ""; position: absolute; inset: 0;
  background-image: linear-gradient(rgba(255,255,255,.05) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,.05) 1px, transparent 1px);
  background-size: 34px 34px; mask-image: linear-gradient(90deg, transparent, #000 60%); pointer-events: none; }
.hero .tag { display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: .14em;
  color: #c9c3ff; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.16);
  padding: 5px 12px; border-radius: 999px; margin-bottom: 12px; }
.hero h1 { color: #fff; font-size: 38px; font-weight: 800; margin: 0 0 6px; line-height: 1.1; }
.hero h1 span { background: linear-gradient(90deg, #8fb8ff, #c4b5fd, #ffd27a);
  -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero p { color: #c8cbe6; font-size: 15.5px; max-width: 760px; margin: 0; line-height: 1.6; }
.hero .chips { margin-top: 18px; display: flex; gap: 10px; flex-wrap: wrap; position: relative; z-index: 2; }
.hero .chip { background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14); border-radius: 12px;
  padding: 8px 14px; font-size: 13px; color: #e6e8ff; }
.hero .chip b { color: #fff; font-size: 15px; margin-right: 4px; }

/* KARTU */
.card { background: #fff; border: 1px solid #e7e9f2; border-radius: 16px; padding: 18px 20px;
  box-shadow: 0 1px 2px rgba(16,24,40,.04); height: 100%; }
.card .lbl { font-size: 12px; font-weight: 700; letter-spacing: .08em; color: #6b7189; text-transform: uppercase; }
.card .big { font-size: 26px; font-weight: 800; color: #111528; margin: 6px 0 2px; line-height: 1.15; }
.card .sub { font-size: 13.5px; color: #555b73; line-height: 1.5; }
.dot { display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin-right: 7px; vertical-align: 1px; }
.winner { border-top: 4px solid var(--c); }
.pill { display: inline-block; font-size: 11.5px; font-weight: 700; border-radius: 999px; padding: 3px 10px; }
.pill.ok { background: #e6f6ee; color: #0f7a46; }
.pill.wait { background: #fff4dc; color: #8a5a00; }
.pill.base { background: #eef0f4; color: #4b5163; }
.note { background: #f4f2ff; border: 1px solid #ddd7ff; border-radius: 14px; padding: 14px 18px;
  color: #2e2763; font-size: 14.5px; line-height: 1.6; }
.warnbox { background: #fff8e8; border: 1px solid #f6dfa6; border-radius: 14px; padding: 14px 18px;
  color: #5c4300; font-size: 14px; line-height: 1.6; }
.model-card h4 { margin: 0 0 4px; font-size: 17px; }
.model-card .maker { font-size: 12.5px; color: #6b7189; margin-bottom: 8px; }

/* TABS */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #e6e8f0; }
.stTabs [data-baseweb="tab"] { padding: 10px 16px; border-radius: 10px 10px 0 0; font-weight: 600; }
.stTabs [aria-selected="true"] { background: #f0eeff; color: #3b2fb0 !important; }
div[data-testid="stMetric"] { background: #fff; border: 1px solid #e7e9f2; border-radius: 14px; padding: 14px 16px; }
</style>
""", unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 420, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, font=dict(family=FONT, size=13, color="#2b3040"),
        margin=dict(l=10, r=10, t=56, b=20), hovermode="x unified" if legend else "closest",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fcfcfb",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0, title=None),
        title=dict(font=dict(size=16, color="#111528")),
        hoverlabel=dict(font=dict(family=FONT)),
    )
    fig.update_xaxes(showgrid=False, linecolor="#d5d8e2")
    fig.update_yaxes(gridcolor="#eceef3", zeroline=False)
    return fig


def color_of(m: str) -> str:
    return MODELS.get(m, {}).get("color", "#999")


# =====================================================================
# DATA
# =====================================================================
@st.cache_data
def load_series() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, usecols=["datetime", "nat_demand", "holiday", "school", "T2M_toc"],
                     parse_dates=["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


@st.cache_data
def load_results(metrics_src, forecasts_src):
    m = pd.read_csv(metrics_src) if metrics_src is not None else pd.DataFrame()
    f = pd.read_csv(forecasts_src, parse_dates=["timestamp"]) if forecasts_src is not None else pd.DataFrame()
    return m, f


with st.sidebar:
    st.markdown("### ⚡ GridCast Panama")
    st.caption("Benchmark model prediksi beban listrik per jam")
    with st.expander("📤 Pakai hasil Colab sendiri (opsional)"):
        up_m = st.file_uploader("metrics.csv", type="csv")
        up_f = st.file_uploader("forecasts.csv", type="csv")

df = load_series()
metrics, fc = load_results(up_m if up_m else (METRICS_PATH if os.path.exists(METRICS_PATH) else None),
                           up_f if up_f else (FORECASTS_PATH if os.path.exists(FORECASTS_PATH) else None))

if metrics.empty:
    st.error("Belum ada file `results/metrics.csv`. Jalankan notebook `notebooks/benchmark_5_model.ipynb` lalu unggah hasilnya.")
    st.stop()

available = [m for m in MODELS if m in set(metrics["model"])] + \
            [m for m in metrics["model"].unique() if m not in MODELS]
missing = [m for m in REQUESTED if m not in available]
horizons = sorted(metrics["horizon_h"].unique())

# ---------------- SIDEBAR KONTROL ----------------
with st.sidebar:
    st.divider()
    h_sel = st.radio("🎯 Horizon prediksi", horizons, index=min(1, len(horizons) - 1),
                     format_func=lambda h: HORIZON_LABEL.get(h, f"{h} jam"),
                     help="Seberapa jauh ke depan model diminta menebak.")
    rank_by = st.radio("📏 Urutkan berdasarkan", ["MAPE (%)", "RMSE"], horizontal=True)
    sel_models = st.multiselect("🧩 Model yang ditampilkan", available, default=available)
    if not sel_models:
        sel_models = available
    st.divider()
    with st.expander("📘 Cara membaca metrik", expanded=False):
        st.markdown("""
**MAPE** — rata-rata *persentase* meleset.
MAPE 5% = rata-rata prediksi meleset 5% dari beban sebenarnya.

**RMSE** — rata-rata meleset dalam **MWh**, tetapi kesalahan besar dihukum lebih berat.
Cocok untuk menilai risiko meleset parah saat beban puncak.

**Cakupan P10–P90** — seberapa sering nilai aktual jatuh di dalam rentang ketidakpastian model.
Idealnya mendekati **80%**.

👉 Untuk semua metrik: **makin kecil makin baik** (kecuali cakupan).
""")

# =====================================================================
# HERO
# =====================================================================
n_req_done = len([m for m in REQUESTED if m in available])
n_hours = f"{len(df):,}".replace(",", ".")
st.markdown(f"""
<div class="hero">
  <div class="tag">TIME-SERIES FOUNDATION MODELS vs STATISTIK KLASIK</div>
  <h1>Prediksi Beban Listrik <span>Panama</span></h1>
  <p>Model mana yang paling jitu menebak kebutuhan listrik nasional Panama 1 hari, 1 minggu, dan 1 bulan ke depan?
  Dashboard ini membandingkan Chronos-2, TimesFM, Moirai, Lag-Llama, dan SARIMAX dengan aturan main yang sama.</p>
  <div class="chips">
    <div class="chip"><b>{n_hours}</b> jam data</div>
    <div class="chip"><b>{df.datetime.min():%b %Y}</b> – <b>{df.datetime.max():%b %Y}</b></div>
    <div class="chip"><b>{n_req_done}/5</b> model sudah dievaluasi</div>
    <div class="chip"><b>{len(horizons)}</b> horizon uji</div>
  </div>
</div>
""", unsafe_allow_html=True)

if missing:
    st.markdown(f"""<div class="warnbox">⏳ <b>Hasil belum lengkap:</b> {", ".join(missing)} belum dijalankan.
    Model-model ini butuh GPU dan unduhan bobot dari Hugging Face, jadi jalankan
    <code>notebooks/benchmark_5_model.ipynb</code> di Google Colab, lalu unggah folder <code>results/</code> ke GitHub.
    Dashboard akan otomatis memperbarui peringkat.</div>""", unsafe_allow_html=True)
    st.write("")

m_h = metrics[(metrics["horizon_h"] == h_sel) & (metrics["model"].isin(sel_models))].sort_values(rank_by)

tab_sum, tab_cmp, tab_fc, tab_err, tab_data, tab_about = st.tabs([
    "🏆 Ringkasan", "⚖️ Perbandingan Model", "🔮 Prediksi vs Aktual",
    "🔬 Analisis Error", "📊 Data Beban Listrik", "📚 Tentang Model & Metode"])

# =====================================================================
# TAB 1 — RINGKASAN
# =====================================================================
with tab_sum:
    st.subheader("Siapa juaranya?")
    st.caption("Juara per horizon berdasarkan MAPE dan RMSE. Kartu berwarna mengikuti warna model di seluruh dashboard.")
    cols = st.columns(len(horizons))
    naive = metrics[metrics["model"] == "Seasonal Naive"].set_index("horizon_h")
    for col, h in zip(cols, horizons):
        g = metrics[(metrics["horizon_h"] == h) & (metrics["model"] != "Seasonal Naive")]
        if g.empty:
            g = metrics[metrics["horizon_h"] == h]
        bm = g.loc[g["MAPE (%)"].idxmin()]
        br = g.loc[g["RMSE"].idxmin()]
        gain = ""
        if h in naive.index:
            d = (1 - bm["MAPE (%)"] / naive.loc[h, "MAPE (%)"]) * 100
            gain = (f"<b>{d:.0f}% lebih akurat</b> dari tebakan naif" if d > 0
                    else f"<b>{-d:.0f}% lebih buruk</b> dari tebakan naif ⚠️")
        same = bm["model"] == br["model"]
        col.markdown(f"""
        <div class="card winner" style="--c:{color_of(bm['model'])}">
          <div class="lbl">Horizon {HORIZON_LABEL.get(h, h)}</div>
          <div class="big"><span class="dot" style="background:{color_of(bm['model'])}"></span>{bm['model']}</div>
          <div class="sub">MAPE <b>{bm['MAPE (%)']:.2f}%</b> · RMSE <b>{bm['RMSE']:.1f} MWh</b></div>
          <div class="sub" style="margin-top:6px">{gain}</div>
          <div class="sub" style="margin-top:6px;color:#6b7189">{'Juara di MAPE dan RMSE' if same else f"Juara RMSE: <b>{br['model']}</b> ({br['RMSE']:.1f} MWh)"}</div>
        </div>""", unsafe_allow_html=True)

    st.write("")
    # Kesimpulan otomatis dalam bahasa sederhana
    best_overall = (metrics[metrics["model"] != "Seasonal Naive"].groupby("model")["Peringkat MAPE"].mean().sort_values())
    top = best_overall.index[0] if len(best_overall) else "-"
    mape_h = metrics[(metrics["horizon_h"] == h_sel)].set_index("model")["MAPE (%)"]
    avg_load = df["nat_demand"].tail(24 * 30).mean()
    best_h = mape_h.idxmin()
    st.markdown(f"""<div class="note">💡 <b>Kesimpulan singkat.</b> Dari model yang sudah dievaluasi, <b>{top}</b> punya
    peringkat MAPE rata-rata terbaik di semua horizon. Untuk horizon <b>{HORIZON_LABEL.get(h_sel)}</b>, model terbaik
    (<b>{best_h}</b>) meleset rata-rata <b>{mape_h.min():.2f}%</b> — kira-kira <b>±{avg_load * mape_h.min() / 100:.0f} MWh</b>
    per jam dari beban rata-rata ~{avg_load:.0f} MWh pada 30 hari terakhir.
    {"<br>⚠️ Peringkat ini <b>sementara</b> karena " + ", ".join(missing) + " belum dijalankan." if missing else ""}</div>""",
                unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns([1.35, 1])
    with c1:
        piv = metrics.pivot_table(index="model", columns="horizon_h", values=rank_by)
        piv = piv.loc[piv.mean(axis=1).sort_values().index]
        fig = go.Figure(go.Heatmap(
            z=piv.values, x=[HORIZON_LABEL.get(c, c) for c in piv.columns], y=piv.index,
            colorscale=[[0, "#1f4f99"], [0.5, "#8fb3e6"], [1, "#eef3fb"]],
            text=np.round(piv.values, 2), texttemplate="%{text}", textfont=dict(size=14),
            hovertemplate="%{y}<br>%{x}<br>" + rank_by + ": %{z:.2f}<extra></extra>",
            xgap=3, ygap=3, colorbar=dict(title=rank_by, thickness=12)))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig, 380, legend=False).update_layout(
            title=f"Peta skor {rank_by} — warna gelap = lebih akurat"), use_container_width=True)
    with c2:
        st.markdown("**Status 5 model yang diminta**")
        for m in REQUESTED + ["Seasonal Naive"]:
            info = MODELS[m]
            if m == "Seasonal Naive":
                pill = '<span class="pill base">Patokan</span>'
            elif m in available:
                pill = '<span class="pill ok">✓ Sudah dievaluasi</span>'
            else:
                pill = '<span class="pill wait">⏳ Menunggu Colab</span>'
            st.markdown(f"""<div style="display:flex;justify-content:space-between;align-items:center;
                padding:9px 4px;border-bottom:1px solid #eef0f5">
                <span><span class="dot" style="background:{info['color']}"></span><b>{m}</b>
                <span style="color:#8a8fa3;font-size:12.5px"> · {info['family']}</span></span>{pill}</div>""",
                        unsafe_allow_html=True)

# =====================================================================
# TAB 2 — PERBANDINGAN
# =====================================================================
with tab_cmp:
    st.subheader(f"Papan peringkat — horizon {HORIZON_LABEL.get(h_sel)}")
    st.caption(f"Diurutkan berdasarkan {rank_by}. Sel hijau = terbaik di kolom itu.")
    show_cols = ["model", "family", "MAPE (%)", "RMSE", "MAE", "Bias", "Cakupan P10-P90 (%)",
                 "Lebih baik dari Naive (MAPE)", "Jumlah Jendela", "Waktu (dtk)"]
    tbl = m_h[[c for c in show_cols if c in m_h.columns]].rename(columns={
        "model": "Model", "family": "Jenis", "Lebih baik dari Naive (MAPE)": "vs Naive (%)",
        "Cakupan P10-P90 (%)": "Cakupan P10–P90 (%)"}).reset_index(drop=True)
    tbl.index = tbl.index + 1
    tbl["Jumlah Jendela"] = tbl["Jumlah Jendela"].astype(int)
    if "Cakupan P10–P90 (%)" in tbl:
        tbl["Cakupan P10–P90 (%)"] = tbl["Cakupan P10–P90 (%)"].map(lambda v: "—" if pd.isna(v) else f"{v:.0f}")
    fmt = {"MAPE (%)": "{:.2f}", "RMSE": "{:.1f}", "MAE": "{:.1f}", "Bias": "{:+.1f}",
           "vs Naive (%)": "{:+.1f}", "Waktu (dtk)": "{:.0f}"}
    st.dataframe(
        tbl.style.format(fmt, na_rep="—").highlight_min(subset=["MAPE (%)", "RMSE", "MAE"], color="#d9f2e4"),
        use_container_width=True)
    st.caption("**Bias** positif = model cenderung menebak terlalu rendah. **vs Naive** = persen perbaikan MAPE "
               "dibanding Seasonal Naive (negatif berarti kalah dari tebakan naif).")

    c1, c2 = st.columns(2)
    for col, met, unit in [(c1, "MAPE (%)", "%"), (c2, "RMSE", " MWh")]:
        d = m_h.sort_values(met, ascending=False)
        fig = go.Figure(go.Bar(
            x=d[met], y=d["model"], orientation="h", marker=dict(color=[color_of(m) for m in d["model"]],
                                                                 cornerradius=4),
            text=[f"{v:.2f}{unit}" if met.startswith("MAPE") else f"{v:.1f}{unit}" for v in d[met]],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
        fig.update_layout(title=f"{met.replace(' (%)', '')} — lebih pendek lebih baik", showlegend=False)
        fig.update_xaxes(title=met, range=[0, d[met].max() * 1.22])
        col.plotly_chart(style_fig(fig, 340, legend=False), use_container_width=True)

    st.markdown("##### Bagaimana akurasi berubah saat menebak makin jauh?")
    d = metrics[metrics["model"].isin(sel_models)].sort_values("horizon_h")
    fig = go.Figure()
    for m in [x for x in MODELS if x in sel_models] + [x for x in sel_models if x not in MODELS]:
        dm = d[d["model"] == m]
        fig.add_trace(go.Scatter(
            x=[HORIZON_LABEL.get(h, h) for h in dm["horizon_h"]], y=dm[rank_by], name=m, mode="lines+markers",
            line=dict(color=color_of(m), width=2, dash="dash" if m == "Seasonal Naive" else "solid"),
            marker=dict(size=9, line=dict(color="#fff", width=2))))
    fig.update_layout(title=f"{rank_by} per horizon", hovermode="x unified")
    fig.update_yaxes(title=rank_by)
    st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        d = m_h.dropna(subset=["Cakupan P10-P90 (%)"])
        if not d.empty:
            fig = go.Figure(go.Bar(x=d["model"], y=d["Cakupan P10-P90 (%)"],
                                   marker=dict(color=[color_of(m) for m in d["model"]], cornerradius=4),
                                   text=[f"{v:.0f}%" for v in d["Cakupan P10-P90 (%)"]], textposition="outside"))
            fig.add_hline(y=80, line_dash="dot", line_color="#555", annotation_text="ideal 80%",
                          annotation_position="top left")
            fig.update_layout(title="Kejujuran rentang ketidakpastian (P10–P90)", showlegend=False)
            fig.update_yaxes(range=[0, 105], title="% nilai aktual di dalam rentang")
            st.plotly_chart(style_fig(fig, 340, legend=False), use_container_width=True)
            st.caption("Di bawah 80% = model terlalu percaya diri; jauh di atas 80% = rentangnya terlalu lebar.")
    with c2:
        fig = go.Figure()
        for _, r in m_h.iterrows():
            fig.add_trace(go.Scatter(x=[max(r["Waktu (dtk)"], 0.1)], y=[r["MAPE (%)"]], name=r["model"],
                                     mode="markers+text", text=[r["model"]], textposition="top center",
                                     marker=dict(size=14, color=color_of(r["model"]),
                                                 line=dict(color="#fff", width=2))))
        fig.update_layout(title="Akurasi vs waktu komputasi (seluruh horizon)", showlegend=False)
        tmin, tmax = max(m_h["Waktu (dtk)"].min(), 0.1), max(m_h["Waktu (dtk)"].max(), 1)
        fig.update_xaxes(type="log", title="Waktu proses total (detik, skala log)",
                         range=[np.log10(tmin) - 0.8, np.log10(tmax) + 0.8])
        lo_, hi_ = m_h["MAPE (%)"].min(), m_h["MAPE (%)"].max()
        pad = max((hi_ - lo_) * 0.35, 0.3)
        fig.update_yaxes(title="MAPE (%) · horizon terpilih", range=[lo_ - pad, hi_ + pad])
        st.plotly_chart(style_fig(fig, 340, legend=False), use_container_width=True)
        st.caption("Kiri-bawah = cepat dan akurat. Waktu foundation model diukur di GPU Colab, SARIMAX di CPU.")

# =====================================================================
# TAB 3 — PREDIKSI vs AKTUAL
# =====================================================================
with tab_fc:
    st.subheader("Lihat langsung tebakan setiap model")
    f_h = fc[(fc["horizon_h"] == h_sel) & (fc["model"].isin(sel_models))]
    wins = sorted(f_h["window"].unique())
    c1, c2, c3 = st.columns([1.4, 1.2, 1])
    with c1:
        if len(wins) > 1:
            starts = f_h.groupby("window")["timestamp"].min()
            w_sel = st.select_slider("Pilih jendela uji", options=wins, value=wins[-1],
                                     format_func=lambda w: f"#{w} · mulai {starts[w]:%d %b %Y %H:%M}")
        else:
            w_sel = wins[0] if wins else 1
            st.info("Horizon ini hanya punya 1 jendela uji (30 hari terakhir data).")
    with c2:
        band_opts = [m for m in sel_models if f_h[f_h["model"] == m]["pred_p10"].notna().any()]
        band_m = st.selectbox("Tampilkan rentang P10–P90 untuk", ["(tidak ada)"] + band_opts,
                              index=1 if band_opts else 0)
    with c3:
        show_ctx = st.toggle("Tampilkan 3 hari sebelumnya", value=True)

    f_w = f_h[f_h["window"] == w_sel]
    fig = go.Figure()
    if show_ctx and not f_w.empty:
        t0 = f_w["timestamp"].min()
        ctx = df[(df["datetime"] < t0) & (df["datetime"] >= t0 - pd.Timedelta(hours=72))]
        fig.add_trace(go.Scatter(x=ctx["datetime"], y=ctx["nat_demand"], name="Riwayat (konteks)",
                                 line=dict(color="#b4b8c5", width=1.5)))
        fig.add_vline(x=t0, line_dash="dot", line_color="#888")
        fig.add_annotation(x=t0, y=1.0, yref="paper", text="mulai prediksi", showarrow=False,
                           xanchor="left", font=dict(size=11, color="#666"))
    if band_m != "(tidak ada)":
        b = f_w[f_w["model"] == band_m]
        fig.add_trace(go.Scatter(x=pd.concat([b["timestamp"], b["timestamp"][::-1]]),
                                 y=pd.concat([b["pred_p90"], b["pred_p10"][::-1]]),
                                 fill="toself", fillcolor=color_of(band_m), opacity=0.16, line=dict(width=0),
                                 name=f"Rentang P10–P90 {band_m}", hoverinfo="skip"))
    act = f_w.drop_duplicates("timestamp")
    fig.add_trace(go.Scatter(x=act["timestamp"], y=act["actual"], name="Aktual",
                             line=dict(color=ACTUAL_COLOR, width=2.6)))
    for m in [x for x in MODELS if x in sel_models] + [x for x in sel_models if x not in MODELS]:
        d = f_w[f_w["model"] == m]
        if d.empty:
            continue
        fig.add_trace(go.Scatter(x=d["timestamp"], y=d["pred_mean"], name=m,
                                 line=dict(color=color_of(m), width=2,
                                           dash="dash" if m == "Seasonal Naive" else "solid")))
    fig.update_yaxes(title="Beban listrik (MWh)")
    fig.update_layout(title=f"Aktual vs prediksi — {HORIZON_LABEL.get(h_sel)}, jendela #{w_sel}")
    st.plotly_chart(style_fig(fig, 520), use_container_width=True)

    # Skor jendela ini
    if not f_w.empty:
        sc = f_w.groupby("model").apply(lambda g: pd.Series({
            "MAPE (%)": (abs(g.actual - g.pred_mean) / g.actual).mean() * 100,
            "RMSE": np.sqrt(((g.actual - g.pred_mean) ** 2).mean())}), include_groups=False).sort_values("MAPE (%)")
        items = list(sc.iterrows())
        for i, (m, r) in enumerate(items):
            if i % 4 == 0:
                cols = st.columns(4)
            col = cols[i % 4]
            col.metric(f"{MODELS.get(m, {}).get('icon', '•')} {m}", f"{r['MAPE (%)']:.2f}%",
                       f"RMSE {r['RMSE']:.0f} MWh", delta_color="off", delta_arrow="off")
    st.download_button("⬇️ Unduh prediksi horizon ini (CSV)", f_h.to_csv(index=False).encode(),
                       file_name=f"prediksi_{h_sel}jam.csv", mime="text/csv")

# =====================================================================
# TAB 4 — ANALISIS ERROR
# =====================================================================
with tab_err:
    st.subheader("Di mana model sering meleset?")
    st.caption("Error = aktual − prediksi. Positif berarti model menebak terlalu rendah.")
    e = fc[(fc["horizon_h"] == h_sel) & (fc["model"].isin(sel_models))].copy()
    e["error"] = e["actual"] - e["pred_mean"]
    e["ape"] = (e["error"].abs() / e["actual"]) * 100
    e["jam"] = e["timestamp"].dt.hour
    order = [x for x in MODELS if x in sel_models] + [x for x in sel_models if x not in MODELS]

    c1, c2 = st.columns(2)
    with c1:
        prof = e.groupby(["model", "jam"])["ape"].mean().reset_index()
        fig = go.Figure()
        for m in order:
            d = prof[prof["model"] == m]
            fig.add_trace(go.Scatter(x=d["jam"], y=d["ape"], name=m, mode="lines",
                                     line=dict(color=color_of(m), width=2,
                                               dash="dash" if m == "Seasonal Naive" else "solid")))
        fig.update_layout(title="Rata-rata % error per jam dalam sehari")
        fig.update_xaxes(title="Jam (waktu Panama)", dtick=3)
        fig.update_yaxes(title="APE rata-rata (%)")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
    with c2:
        fig = go.Figure()
        for m in order:
            d = e[e["model"] == m]
            fig.add_trace(go.Box(y=d["error"], name=m, marker_color=color_of(m), boxmean=True,
                                 line=dict(width=1.5)))
        fig.add_hline(y=0, line_color="#444", line_width=1)
        fig.update_layout(title="Sebaran error (MWh) — kotak sempit & di sekitar 0 = bagus", showlegend=False)
        fig.update_yaxes(title="Error (MWh)")
        st.plotly_chart(style_fig(fig, 380, legend=False), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        lo, hi = e["actual"].min(), e["actual"].max()
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Prediksi sempurna",
                                 line=dict(color="#666", dash="dot", width=1.5)))
        for m in order:
            d = e[e["model"] == m]
            fig.add_trace(go.Scatter(x=d["actual"], y=d["pred_mean"], mode="markers", name=m,
                                     marker=dict(color=color_of(m), size=8, opacity=0.6,
                                                 line=dict(color="#fff", width=1))))
        fig.update_layout(title="Aktual vs prediksi — makin dekat garis makin baik", hovermode="closest")
        fig.update_xaxes(title="Aktual (MWh)")
        fig.update_yaxes(title="Prediksi (MWh)")
        st.plotly_chart(style_fig(fig, 400), use_container_width=True)
    with c2:
        fig = go.Figure()
        for m in order:
            d = e[e["model"] == m].groupby("step")["ape"].mean()
            if len(d) > 48:
                d = d.rolling(24, min_periods=1).mean()
            fig.add_trace(go.Scatter(x=d.index, y=d.values, name=m,
                                     line=dict(color=color_of(m), width=2,
                                               dash="dash" if m == "Seasonal Naive" else "solid")))
        fig.update_layout(title="Error makin besar saat menebak makin jauh?")
        fig.update_xaxes(title="Langkah ke depan (jam)")
        fig.update_yaxes(title="APE rata-rata (%)" + (" · rata-rata bergerak 24 jam" if h_sel > 48 else ""))
        st.plotly_chart(style_fig(fig, 400), use_container_width=True)

# =====================================================================
# TAB 5 — DATA
# =====================================================================
with tab_data:
    st.subheader("Mengenal datanya dulu")
    y = df["nat_demand"]
    peak_row = df.loc[y.idxmax()]
    c = st.columns(4)
    c[0].metric("Rata-rata beban", f"{y.mean():,.0f} MWh")
    c[1].metric("Beban puncak", f"{y.max():,.0f} MWh", f"{peak_row.datetime:%d %b %Y, %H:%M}", delta_color="off", delta_arrow="off")
    c[2].metric("Beban terendah", f"{y.min():,.0f} MWh", f"{df.loc[y.idxmin(), 'datetime']:%d %b %Y} (blackout)", delta_color="off", delta_arrow="off")
    m19 = df[(df.datetime.dt.year == 2019) & df.datetime.dt.month.isin([4, 5, 6])]["nat_demand"].mean()
    m20 = df[(df.datetime.dt.year == 2020) & df.datetime.dt.month.isin([4, 5, 6])]["nat_demand"].mean()
    c[3].metric("Dampak COVID-19", f"{(m20 / m19 - 1) * 100:+.1f}%", "Apr–Jun 2020 vs 2019", delta_color="off", delta_arrow="off")

    daily = df.set_index("datetime")["nat_demand"].resample("D").mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.index, y=daily.values, name="Rata-rata harian",
                             line=dict(color="#9db7e0", width=1)))
    fig.add_trace(go.Scatter(x=daily.index, y=daily.rolling(30).mean(), name="Rata-rata bergerak 30 hari",
                             line=dict(color="#1f4f99", width=2.5)))
    fig.add_vrect(x0=pd.Timestamp("2020-03-15"), x1=daily.index.max(), fillcolor="#eda100", opacity=0.12, line_width=0,
                  annotation_text="COVID-19", annotation_position="top left")
    t_start = df["datetime"].iloc[-720]
    fig.add_vrect(x0=t_start, x1=daily.index.max(), fillcolor="#6250d6", opacity=0.18, line_width=0,
                  annotation_text="periode uji", annotation_position="bottom left")
    fig.update_layout(title="Tren beban listrik nasional Panama 2015–2020")
    fig.update_yaxes(title="MWh")
    st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        hm = df.assign(h=df.datetime.dt.hour, d=df.datetime.dt.dayofweek).pivot_table(
            index="d", columns="h", values="nat_demand", aggfunc="mean")
        hm.index = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]
        fig = go.Figure(go.Heatmap(z=hm.values, x=hm.columns, y=hm.index, xgap=1, ygap=1,
                                   colorscale=[[0, "#eef3fb"], [0.5, "#8fb3e6"], [1, "#1f4f99"]],
                                   colorbar=dict(title="MWh", thickness=12),
                                   hovertemplate="%{y}, jam %{x}:00<br>%{z:.0f} MWh<extra></extra>"))
        fig.update_layout(title="Pola mingguan: kapan listrik paling banyak dipakai?")
        fig.update_xaxes(title="Jam", dtick=3)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig, 360, legend=False), use_container_width=True)
    with c2:
        dd = df.assign(jam=df.datetime.dt.hour,
                       tipe=np.where(df.holiday == 1, "Hari libur",
                                     np.where(df.datetime.dt.dayofweek >= 5, "Akhir pekan", "Hari kerja")))
        prof = dd.groupby(["tipe", "jam"])["nat_demand"].mean().reset_index()
        fig = go.Figure()
        for t, colr in [("Hari kerja", "#2a78d6"), ("Akhir pekan", "#eb6834"), ("Hari libur", "#1baf7a")]:
            d = prof[prof.tipe == t]
            fig.add_trace(go.Scatter(x=d.jam, y=d.nat_demand, name=t, line=dict(color=colr, width=2)))
        fig.update_layout(title="Profil harian: hari kerja vs akhir pekan vs libur")
        fig.update_xaxes(title="Jam", dtick=3)
        fig.update_yaxes(title="MWh")
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        yr = df.assign(tahun=df.datetime.dt.year.astype(str))
        fig = px.box(yr, x="tahun", y="nat_demand", points=False, color_discrete_sequence=["#2a78d6"])
        fig.update_layout(title="Sebaran beban per tahun")
        fig.update_yaxes(title="MWh")
        st.plotly_chart(style_fig(fig, 340, legend=False), use_container_width=True)
    with c2:
        sm = df.sample(min(8000, len(df)), random_state=1)
        fig = go.Figure(go.Scattergl(x=sm["T2M_toc"], y=sm["nat_demand"], mode="markers",
                                     marker=dict(size=5, color="#2a78d6", opacity=0.25)))
        fig.update_layout(title="Makin panas, makin banyak AC menyala (suhu Tocumen)")
        fig.update_xaxes(title="Suhu °C")
        fig.update_yaxes(title="MWh")
        st.plotly_chart(style_fig(fig, 340, legend=False), use_container_width=True)

# =====================================================================
# TAB 6 — TENTANG
# =====================================================================
with tab_about:
    st.subheader("Kenalan dengan para model")
    cols = st.columns(3)
    for i, (m, info) in enumerate(MODELS.items()):
        status = ('<span class="pill base">Patokan</span>' if m == "Seasonal Naive" else
                  '<span class="pill ok">✓ Ada hasil</span>' if m in available else
                  '<span class="pill wait">⏳ Belum dijalankan</span>')
        cfg = metrics.loc[metrics["model"] == m, "config"].dropna()
        cfg_txt = f'<div class="sub" style="margin-top:8px;font-size:12px;color:#8a8fa3">⚙️ {cfg.iloc[0]}</div>' \
            if len(cfg) and str(cfg.iloc[0]) not in ("", "nan") else ""
        cols[i % 3].markdown(f"""<div class="card model-card" style="border-top:4px solid {info['color']};margin-bottom:14px">
            <h4>{info['icon']} {m}</h4><div class="maker">{info['maker']} · {info['family']} &nbsp; {status}</div>
            <div class="sub">{info['short']}</div>{cfg_txt}</div>""", unsafe_allow_html=True)

    st.subheader("Aturan main evaluasi")
    st.markdown("""
Semua model diuji dengan cara yang **sama persis** supaya adil:
- Setiap model hanya boleh melihat data **sebelum** periode uji dimulai — tidak ada kebocoran data masa depan.
- Model foundation (Chronos-2, TimesFM, Moirai, Lag-Llama) dipakai **zero-shot**: tidak dilatih ulang, hanya diberi riwayat beban sebagai konteks.
- SARIMAX boleh memakai **kalender** (hari libur, masa sekolah) karena informasi ini sudah diketahui sebelumnya. Cuaca tidak dipakai, karena cuaca masa depan tidak diketahui saat memprediksi.
- Prediksi titik = **median** (P50) untuk model probabilistik.
""")
    last = df["datetime"].max()
    rows = []
    for h, n in [(24, 7), (168, 4), (720, 1)]:
        for k in range(n):
            end = last - pd.Timedelta(hours=k * h) + pd.Timedelta(hours=1)
            rows.append(dict(Horizon=HORIZON_LABEL[h], mulai=end - pd.Timedelta(hours=h), selesai=end,
                             Jendela=f"#{n - k}"))
    tl = pd.DataFrame(rows)
    fig = px.timeline(tl, x_start="mulai", x_end="selesai", y="Horizon", color="Horizon", text="Jendela",
                      color_discrete_sequence=["#2a78d6", "#6250d6", "#1baf7a"])
    fig.update_traces(marker_line_color="#fff", marker_line_width=2, textfont_color="#fff")
    fig.update_layout(title="Jendela uji: 7×24 jam, 4×168 jam, 1×720 jam di akhir data", showlegend=False)
    st.plotly_chart(style_fig(fig, 260, legend=False), use_container_width=True)
    st.caption("Sumber data: beban listrik nasional Panama per jam beserta data cuaca 3 kota dan kalender (libur & sekolah), Jan 2015 – Jun 2020.")
