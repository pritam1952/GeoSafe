"""
GeoSafe — Road Accident Hotspot Prediction System
Run locally:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="GeoSafe AI — Road Safety Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------
# Global CSS — deep navy + signal-amber design system
# ---------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.main { background: #070d1a; }
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1300px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0c1629;
    border-right: 1px solid rgba(255,200,0,0.12);
}
section[data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
    font-size: 14px;
    font-weight: 500;
    padding: 6px 0;
    transition: color 0.2s;
}
section[data-testid="stSidebar"] .stRadio label:hover { color: #f59e0b !important; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0c1629 0%, #0f2044 40%, #1a1000 100%);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 20px;
    padding: 48px 56px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(245,158,11,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #f59e0b;
    margin-bottom: 12px;
}
.hero-title {
    font-size: 52px;
    font-weight: 900;
    line-height: 1.05;
    color: #ffffff;
    margin: 0 0 14px 0;
}
.hero-title span { color: #f59e0b; }
.hero-subtitle {
    font-size: 16px;
    color: #94a3b8;
    max-width: 560px;
    line-height: 1.6;
    margin: 0;
}

/* ── Stat Cards ── */
.stat-grid { display: flex; gap: 16px; margin-bottom: 2rem; }
.stat-card {
    flex: 1;
    background: #0c1629;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 24px 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s, transform 0.25s;
}
.stat-card:hover {
    border-color: rgba(245,158,11,0.35);
    transform: translateY(-2px);
}
.stat-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 0 0 14px 14px;
}
.stat-card.amber::after  { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.stat-card.red::after    { background: linear-gradient(90deg, #ef4444, #f87171); }
.stat-card.blue::after   { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
.stat-card.green::after  { background: linear-gradient(90deg, #22c55e, #4ade80); }
.stat-icon {
    font-size: 28px;
    margin-bottom: 12px;
    display: block;
}
.stat-value {
    font-size: 36px;
    font-weight: 800;
    color: #fff;
    line-height: 1;
    font-family: 'JetBrains Mono', monospace;
}
.stat-label {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #475569;
    margin-top: 6px;
}

/* ── Section headings ── */
.section-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #f59e0b;
    margin-bottom: 8px;
}
.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 20px;
}

/* ── Predict form ── */
.form-card {
    background: #0c1629;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
}
.form-card h4 {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #f59e0b;
    margin: 0 0 18px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(245,158,11,0.15);
}

/* ── Result cards ── */
.result-banner {
    border-radius: 16px;
    padding: 28px 32px;
    margin-top: 1.5rem;
    border: 1px solid;
}
.result-fatal  { background: rgba(239,68,68,0.08);  border-color: rgba(239,68,68,0.3); }
.result-major  { background: rgba(251,146,60,0.08); border-color: rgba(251,146,60,0.3); }
.result-minor  { background: rgba(34,197,94,0.08);  border-color: rgba(34,197,94,0.3); }
.result-title  { font-size: 13px; color: #94a3b8; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 6px; }
.result-value  { font-size: 40px; font-weight: 900; font-family: 'JetBrains Mono', monospace; }
.result-fatal  .result-value { color: #f87171; }
.result-major  .result-value { color: #fb923c; }
.result-minor  .result-value { color: #4ade80; }

/* ── Risk reason pills ── */
.reason-pill {
    display: inline-block;
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 13px;
    color: #fcd34d;
    margin: 5px 5px 0 0;
    line-height: 1.4;
}

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #f59e0b, #ef4444) !important;
    border-radius: 99px !important;
}
.stProgress > div > div > div {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 99px !important;
}

/* ── Sidebar brand ── */
.sidebar-brand {
    text-align: center;
    padding: 20px 10px 30px;
}
.sidebar-brand-title {
    font-size: 26px;
    font-weight: 900;
    color: #fff;
    letter-spacing: -0.5px;
}
.sidebar-brand-title span { color: #f59e0b; }
.sidebar-brand-sub {
    font-size: 11px;
    color: #475569;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 4px;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 0 0 20px;
}

/* ── Plotly chart bg ── */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }

/* ── About timeline ── */
.timeline-step {
    display: flex;
    gap: 18px;
    margin-bottom: 22px;
    align-items: flex-start;
}
.timeline-dot {
    width: 36px; height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #f59e0b, #b45309);
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 700; color: #000;
    flex-shrink: 0;
}
.timeline-body h4 { font-size: 15px; font-weight: 700; color: #e2e8f0; margin: 0 0 4px; }
.timeline-body p  { font-size: 13px; color: #64748b; margin: 0; line-height: 1.5; }

/* ── Table ── */
.dataframe thead th {
    background: #0f2044 !important;
    color: #f59e0b !important;
    font-size: 12px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}
.dataframe tbody tr:hover td { background: rgba(245,158,11,0.05) !important; }

/* ── Misc ── */
h1,h2,h3 { color: #f1f5f9 !important; }
.stSelectbox label, .stSlider label, .stCheckbox label, .stMultiselect label {
    color: #94a3b8 !important; font-size: 13px !important; font-weight: 500 !important;
}
div[data-testid="stMetricValue"] { color: #f1f5f9; font-weight: 800; }
div[data-testid="stMetricLabel"] { color: #475569 !important; font-size: 12px !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------
DATA_DIR = Path("data")
if not DATA_DIR.exists():
    DATA_DIR = Path("../data")

FEATURE_ORDER = [
    "lanes", "traffic_signal", "temperature", "vehicles_involved",
    "casualties", "is_peak_hour", "is_weekend", "hour", "is_night",
    "city", "state", "road_type", "weather", "visibility",
    "traffic_density", "cause", "day_of_week",
]

SEVERITY_COLOR = {"fatal": "#ef4444", "major": "#f97316", "minor": "#22c55e"}

# ---------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model    = joblib.load(DATA_DIR / "severity_model.pkl")
    encoders = joblib.load(DATA_DIR / "label_encoders.pkl")
    target_le = joblib.load(DATA_DIR / "target_encoder.pkl")
    return model, encoders, target_le

@st.cache_data
def load_data():
    return pd.read_csv(DATA_DIR / "cleaned_accidents.csv")

@st.cache_data
def load_hotspots():
    return pd.read_csv(DATA_DIR / "hotspots.csv")

try:
    model, encoders, target_le = load_artifacts()
    df       = load_data()
    hotspots = load_hotspots()
    LOAD_ERROR = None
except FileNotFoundError as e:
    LOAD_ERROR = str(e)


# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">Geo<span>Safe</span></div>
        <div class="sidebar-brand-sub">Road Safety Intelligence</div>
    </div>
    <hr class="sidebar-divider"/>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Dashboard", "🗺️  Hotspot Map", "⚠️  Severity Predictor", "ℹ️  About"],
        label_visibility="collapsed",
    )

    st.markdown("<br/>", unsafe_allow_html=True)

    if LOAD_ERROR is None:
        st.markdown("""
        <div style='background:#0f2044;border-radius:10px;padding:14px 16px;'>
            <div style='font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#f59e0b;margin-bottom:8px;font-weight:700;'>System Status</div>
            <div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>
                <span style='width:8px;height:8px;background:#22c55e;border-radius:50%;display:inline-block;'></span>
                <span style='font-size:12px;color:#94a3b8;'>Model loaded</span>
            </div>
            <div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>
                <span style='width:8px;height:8px;background:#22c55e;border-radius:50%;display:inline-block;'></span>
                <span style='font-size:12px;color:#94a3b8;'>Dataset ready</span>
            </div>
            <div style='display:flex;align-items:center;gap:8px;'>
                <span style='width:8px;height:8px;background:#22c55e;border-radius:50%;display:inline-block;'></span>
                <span style='font-size:12px;color:#94a3b8;'>Hotspots indexed</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("⚠️ Data files missing")

    st.markdown("""
    <div style='margin-top:auto;padding-top:30px;font-size:11px;color:#1e293b;text-align:center;'>
        GeoSafe v2.0 · XGBoost + DBSCAN
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------
# Load-error guard
# ---------------------------------------------------------------
if LOAD_ERROR:
    st.error(
        f"**Couldn't load required files from** `{DATA_DIR.resolve()}`\n\n"
        f"Missing: `{LOAD_ERROR}`\n\n"
        "Run `Hotspot_Detection.ipynb` and `Model_Training.ipynb` first so "
        "`cleaned_accidents.csv`, `hotspots.csv`, `severity_model.pkl`, "
        "`label_encoders.pkl`, and `target_encoder.pkl` exist in `data/`."
    )
    st.stop()


# =================================================================
# DASHBOARD
# =================================================================
if page == "🏠  Dashboard":

    # Hero
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-eyebrow">AI-Powered Road Safety Platform</div>
        <div class="hero-title">Predict. <span>Prevent.</span> Protect.</div>
        <p class="hero-subtitle">
            GeoSafe analyses road, weather, and traffic conditions across India to
            surface accident hotspots and forecast severity — before incidents happen.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI cards
    total     = len(df)
    n_spots   = len(hotspots)
    high_risk = int((hotspots["risk_level"] == "High").sum())
    fatal_pct = round((df["accident_severity"] == "fatal").mean() * 100, 1) if "accident_severity" in df.columns else "—"

    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card amber">
            <span class="stat-icon">🚗</span>
            <div class="stat-value">{total:,}</div>
            <div class="stat-label">Total Accidents</div>
        </div>
        <div class="stat-card blue">
            <span class="stat-icon">📍</span>
            <div class="stat-value">{n_spots:,}</div>
            <div class="stat-label">Hotspot Clusters</div>
        </div>
        <div class="stat-card red">
            <span class="stat-icon">🔴</span>
            <div class="stat-value">{high_risk:,}</div>
            <div class="stat-label">High-Risk Zones</div>
        </div>
        <div class="stat-card green">
            <span class="stat-icon">⚠️</span>
            <div class="stat-value">{fatal_pct}%</div>
            <div class="stat-label">Fatal Rate</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Charts row 1
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-label">Breakdown</div><div class="section-title">Severity Distribution</div>', unsafe_allow_html=True)
        sev_counts = df["accident_severity"].value_counts().reset_index()
        sev_counts.columns = ["severity", "count"]
        colors = [SEVERITY_COLOR.get(s, "#94a3b8") for s in sev_counts["severity"]]
        fig_sev = go.Figure(go.Bar(
            x=sev_counts["severity"].str.upper(),
            y=sev_counts["count"],
            marker_color=colors,
            marker_line_width=0,
            text=sev_counts["count"],
            textfont=dict(color="#fff", size=12, family="JetBrains Mono"),
            textposition="outside",
        ))
        fig_sev.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter"),
            margin=dict(t=10, b=20, l=10, r=10),
            xaxis=dict(showgrid=False, tickfont=dict(size=12, color="#94a3b8")),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=11)),
            showlegend=False,
            height=320,
        )
        st.plotly_chart(fig_sev, use_container_width=True, config={"displayModeBar": False})

    with c2:
        st.markdown('<div class="section-label">Location</div><div class="section-title">Top 10 Cities by Accidents</div>', unsafe_allow_html=True)
        city_counts = df["city"].value_counts().head(10).reset_index()
        city_counts.columns = ["city", "count"]
        fig_city = go.Figure(go.Bar(
            x=city_counts["count"],
            y=city_counts["city"],
            orientation="h",
            marker=dict(
                color=city_counts["count"],
                colorscale=[[0, "#1e293b"], [1, "#f59e0b"]],
                line_width=0,
            ),
            text=city_counts["count"],
            textfont=dict(color="#fff", size=11, family="JetBrains Mono"),
            textposition="outside",
        ))
        fig_city.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter"),
            margin=dict(t=10, b=20, l=10, r=30),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(showgrid=False, tickfont=dict(size=11, color="#cbd5e1")),
            height=320,
        )
        st.plotly_chart(fig_city, use_container_width=True, config={"displayModeBar": False})

    # Charts row 2
    c3, c4 = st.columns(2)

    with c3:
        if "hour" in df.columns:
            st.markdown('<div class="section-label">Temporal</div><div class="section-title">Accidents by Hour of Day</div>', unsafe_allow_html=True)
            hour_counts = df["hour"].value_counts().sort_index().reset_index()
            hour_counts.columns = ["hour", "count"]
            fig_hour = go.Figure(go.Scatter(
                x=hour_counts["hour"],
                y=hour_counts["count"],
                mode="lines",
                fill="tozeroy",
                line=dict(color="#f59e0b", width=2.5),
                fillcolor="rgba(245,158,11,0.12)",
            ))
            fig_hour.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Inter"),
                margin=dict(t=10, b=20, l=10, r=10),
                xaxis=dict(showgrid=False, title="Hour", tickfont=dict(size=11)),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=11)),
                height=260,
            )
            st.plotly_chart(fig_hour, use_container_width=True, config={"displayModeBar": False})

    with c4:
        if "weather" in df.columns:
            st.markdown('<div class="section-label">Conditions</div><div class="section-title">Accidents by Weather</div>', unsafe_allow_html=True)
            weather_counts = df["weather"].value_counts().reset_index()
            weather_counts.columns = ["weather", "count"]
            fig_wx = go.Figure(go.Pie(
                labels=weather_counts["weather"],
                values=weather_counts["count"],
                hole=0.52,
                marker=dict(colors=["#f59e0b","#3b82f6","#ef4444","#22c55e","#8b5cf6","#ec4899"]),
                textfont=dict(size=11, color="#fff"),
                showlegend=True,
            ))
            fig_wx.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Inter"),
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(font=dict(size=11, color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
                height=260,
            )
            st.plotly_chart(fig_wx, use_container_width=True, config={"displayModeBar": False})

    # Risk overview table
    st.markdown('<div class="section-label">Hotspot Intelligence</div><div class="section-title">Top 10 High-Risk Zones</div>', unsafe_allow_html=True)
    top_spots = hotspots.sort_values("avg_risk_score", ascending=False).head(10)[
        ["city", "risk_level", "accident_count", "avg_risk_score"]
    ].copy()
    top_spots.columns = ["City", "Risk Level", "Accidents", "Avg Risk Score"]
    top_spots["Avg Risk Score"] = top_spots["Avg Risk Score"].round(3)
    st.dataframe(top_spots, use_container_width=True, hide_index=True)


# =================================================================
# HOTSPOT MAP
# =================================================================
elif page == "🗺️  Hotspot Map":

    st.markdown("""
    <div class="hero-eyebrow">Live Intelligence Layer</div>
    <div class="section-title" style="font-size:36px;font-weight:900;color:#f1f5f9;">Accident Hotspot Map</div>
    """, unsafe_allow_html=True)

    # Filter + summary row
    col_f, col_s1, col_s2, col_s3 = st.columns([3, 1, 1, 1])
    with col_f:
        risk_filter = st.multiselect(
            "Filter by risk level",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )

    filtered = hotspots[hotspots["risk_level"].isin(risk_filter)]
    with col_s1:
        st.metric("Shown", len(filtered))
    with col_s2:
        st.metric("High Risk", int((filtered["risk_level"] == "High").sum()))
    with col_s3:
        st.metric("Total Accidents", int(filtered["accident_count"].sum()))

    # Folium map
    color_map = {"High": "red", "Medium": "orange", "Low": "green"}
    m = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=5,
        tiles="CartoDB dark_matter",
    )

    for _, row in filtered.iterrows():
        radius = 5 + min(row["accident_count"] / 4, 14)
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color_map[row["risk_level"]],
            fill=True,
            fill_color=color_map[row["risk_level"]],
            fill_opacity=0.65,
            weight=1.2,
            popup=folium.Popup(
                f"""
                <div style='font-family:Inter,sans-serif;min-width:180px;'>
                    <b style='font-size:14px;'>{row['city']}</b><br/>
                    <hr style='margin:5px 0;border-color:#ddd;'/>
                    <span style='color:#666;font-size:12px;'>Risk Level</span><br/>
                    <span style='font-size:13px;font-weight:700;
                        color:{"#dc2626" if row["risk_level"]=="High" else "#ea580c" if row["risk_level"]=="Medium" else "#16a34a"};'>
                        {row["risk_level"]}
                    </span><br/><br/>
                    <span style='color:#666;font-size:12px;'>Accidents Recorded</span><br/>
                    <b style='font-size:15px;'>{row['accident_count']}</b><br/><br/>
                    <span style='color:#666;font-size:12px;'>Avg Risk Score</span><br/>
                    <b style='font-size:15px;'>{row['avg_risk_score']:.3f}</b>
                </div>
                """,
                max_width=220,
            ),
            tooltip=f"{row['city']} — {row['risk_level']} Risk",
        ).add_to(m)

    st_folium(m, width="100%", height=560)

    # Sortable table below map
    st.markdown('<div class="section-label" style="margin-top:1.5rem;">Detailed View</div>', unsafe_allow_html=True)
    display_df = filtered[["city", "risk_level", "accident_count", "avg_risk_score"]].sort_values(
        "avg_risk_score", ascending=False
    ).copy()
    display_df.columns = ["City", "Risk Level", "Accidents", "Avg Risk Score"]
    display_df["Avg Risk Score"] = display_df["Avg Risk Score"].round(3)
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# =================================================================
# SEVERITY PREDICTOR
# =================================================================
elif page == "⚠️  Severity Predictor":

    st.markdown("""
    <div class="hero-eyebrow">Predictive Engine</div>
    <div class="section-title" style="font-size:36px;font-weight:900;color:#f1f5f9;">Accident Severity Predictor</div>
    <p style="color:#64748b;font-size:14px;margin-bottom:1.5rem;">
        Enter pre-accident conditions. The model returns predicted severity and a risk score
        using XGBoost trained on Indian road accident data.
    </p>
    """, unsafe_allow_html=True)

    with st.form("predict_form"):
        # Location & Road
        st.markdown('<div class="form-card"><h4>📍 Location & Road Conditions</h4>', unsafe_allow_html=True)
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            city = st.selectbox("City", sorted(df["city"].unique()))
        state = df.loc[df["city"] == city, "state"].iloc[0]
        with r1c2:
            road_type = st.selectbox("Road Type", sorted(df["road_type"].unique()))
        with r1c3:
            lanes = st.slider("Number of Lanes", 1, int(df["lanes"].max()), 2)
        st.markdown('</div>', unsafe_allow_html=True)

        # Environment
        st.markdown('<div class="form-card"><h4>🌦️ Environmental Conditions</h4>', unsafe_allow_html=True)
        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            weather = st.selectbox("Weather", sorted(df["weather"].unique()))
        with r2c2:
            visibility = st.selectbox("Visibility", sorted(df["visibility"].unique()))
        with r2c3:
            temperature = st.slider(
                "Temperature (°C)",
                int(df["temperature"].min()),
                int(df["temperature"].max()),
                28,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # Traffic & Time
        st.markdown('<div class="form-card"><h4>🚦 Traffic & Time</h4>', unsafe_allow_html=True)
        r3c1, r3c2, r3c3, r3c4 = st.columns(4)
        with r3c1:
            traffic_density = st.selectbox("Traffic Density", sorted(df["traffic_density"].unique()))
        with r3c2:
            day_of_week = st.selectbox("Day of Week", sorted(df["day_of_week"].unique()))
        with r3c3:
            hour = st.slider("Hour (24h)", 0, 23, 19)
        with r3c4:
            traffic_signal = st.checkbox("Traffic Signal Present", value=False)

        cause = st.selectbox("Likely Cause of Accident", sorted(df["cause"].unique()))
        st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button(
            "⚡  Run Severity Prediction",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        is_weekend  = 1 if day_of_week in ["Saturday", "Sunday"] else 0
        is_peak_hour = 1 if hour in [8, 9, 10, 17, 18, 19, 20] else 0
        is_night    = 1 if (hour >= 20 or hour <= 5) else 0

        input_dict = {
            "lanes": lanes,
            "traffic_signal": int(traffic_signal),
            "temperature": temperature,
            "vehicles_involved": df["vehicles_involved"].median(),
            "casualties": df["casualties"].median(),
            "is_peak_hour": is_peak_hour,
            "is_weekend": is_weekend,
            "hour": hour,
            "is_night": is_night,
            "city": city,
            "state": state,
            "road_type": road_type,
            "weather": weather,
            "visibility": visibility,
            "traffic_density": traffic_density,
            "cause": cause,
            "day_of_week": day_of_week,
        }

        input_df = pd.DataFrame([input_dict])
        for col, le in encoders.items():
            input_df[col] = le.transform(input_df[col])
        input_df = input_df[FEATURE_ORDER]

        pred_class = model.predict(input_df)[0]
        pred_label = target_le.inverse_transform([pred_class])[0]

        proba   = model.predict_proba(input_df)[0]
        classes = list(target_le.classes_)
        fatal_p = proba[classes.index("fatal")] if "fatal" in classes else 0
        major_p = proba[classes.index("major")] if "major" in classes else 0
        risk_score = min(fatal_p + 0.5 * major_p, 1.0)

        result_class = pred_label  # fatal / major / minor
        emoji_map = {"fatal": "🔴", "major": "🟠", "minor": "🟢"}

        # Result banner
        st.markdown(f"""
        <div class="result-banner result-{result_class}">
            <div style="display:flex;gap:48px;align-items:center;flex-wrap:wrap;">
                <div>
                    <div class="result-title">Predicted Severity</div>
                    <div class="result-value">{emoji_map[result_class]} {pred_label.upper()}</div>
                </div>
                <div>
                    <div class="result-title">Composite Risk Score</div>
                    <div class="result-value">{risk_score:.3f}</div>
                </div>
                <div style="flex:1;min-width:200px;">
                    <div class="result-title" style="margin-bottom:10px;">Confidence Breakdown</div>
        """, unsafe_allow_html=True)

        # Mini probability chart
        prob_labels = [c.upper() for c in classes]
        prob_vals   = [round(p * 100, 1) for p in proba]
        bar_colors  = [SEVERITY_COLOR.get(c, "#94a3b8") for c in classes]
        fig_prob = go.Figure(go.Bar(
            x=prob_labels, y=prob_vals,
            marker_color=bar_colors,
            marker_line_width=0,
            text=[f"{v}%" for v in prob_vals],
            textfont=dict(color="#fff", size=11),
            textposition="outside",
        ))
        fig_prob.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter"),
            margin=dict(t=10, b=5, l=5, r=5),
            height=150,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False, showticklabels=False, range=[0, max(prob_vals) * 1.35]),
            showlegend=False,
        )
        st.plotly_chart(fig_prob, use_container_width=True, config={"displayModeBar": False})

        st.markdown("</div></div></div>", unsafe_allow_html=True)

        # Risk score gauge
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Risk Gauge</div>', unsafe_allow_html=True)
        gauge_color = "#ef4444" if risk_score > 0.65 else "#f97316" if risk_score > 0.35 else "#22c55e"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(risk_score * 100, 1),
            number={"suffix": "%", "font": {"size": 40, "color": "#fff", "family": "JetBrains Mono"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#334155", "tickfont": {"color": "#475569"}},
                "bar": {"color": gauge_color, "thickness": 0.25},
                "bgcolor": "#0c1629",
                "bordercolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0,  35], "color": "rgba(34,197,94,0.1)"},
                    {"range": [35, 65], "color": "rgba(249,115,22,0.1)"},
                    {"range": [65,100], "color": "rgba(239,68,68,0.1)"},
                ],
                "threshold": {
                    "line": {"color": gauge_color, "width": 3},
                    "thickness": 0.8,
                    "value": round(risk_score * 100, 1),
                },
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#94a3b8", "family": "Inter"},
            height=240,
            margin=dict(t=10, b=10, l=20, r=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

        # Risk factors
        reasons = []
        if traffic_density == "high":
            reasons.append("🚗  High traffic density increases collision probability")
        if weather in ["rain", "Rain"]:
            reasons.append("🌧️  Rain reduces traction and braking distance")
        if weather in ["fog", "Fog"]:
            reasons.append("🌫️  Fog critically reduces sight lines")
        if is_peak_hour:
            reasons.append(f"⏱️  Hour {hour}:00 falls in high-volume peak period")
        if is_night:
            reasons.append("🌙  Night-time conditions reduce driver reaction time")
        if road_type in ["highway", "Highway"]:
            reasons.append("🛣️  Highways carry higher-speed impacts")
        if not traffic_signal:
            reasons.append("🚦  No traffic signal — uncontrolled intersection risk")
        if not reasons:
            reasons.append("✅  Conditions are within a typical / average risk range")

        st.markdown('<div class="section-label" style="margin-top:0.5rem;">Risk Factors</div>', unsafe_allow_html=True)
        pills = "".join([f'<span class="reason-pill">{r}</span>' for r in reasons])
        st.markdown(f'<div style="margin-bottom:1.5rem;">{pills}</div>', unsafe_allow_html=True)


# =================================================================
# ABOUT
# =================================================================
else:
    st.markdown("""
    <div class="hero-eyebrow">Project Overview</div>
    <div class="section-title" style="font-size:36px;font-weight:900;color:#f1f5f9;">About GeoSafe</div>
    <p style="color:#64748b;font-size:15px;max-width:700px;line-height:1.7;margin-bottom:2rem;">
        GeoSafe is an end-to-end AI platform built to surface road accident hotspots
        across India and predict crash severity before an incident occurs — giving
        planners, traffic authorities, and commuters actionable intelligence.
    </p>
    """, unsafe_allow_html=True)

    col_pipe, col_stack = st.columns(2)

    with col_pipe:
        st.markdown('<div class="section-label">Methodology</div><div class="section-title">Pipeline</div>', unsafe_allow_html=True)
        steps = [
            ("1", "Data Collection", "Indian road-accident records — location, conditions, time, severity."),
            ("2", "Preprocessing", "Null handling, outlier removal, categorical encoding, feature derivation."),
            ("3", "EDA & Feature Engineering", "Hour buckets, peak flags, night flag, day-of-week extracted."),
            ("4", "Hotspot Detection", "DBSCAN clustering identifies geographic accident density zones."),
            ("5", "Model Training", "XGBoost classifier trained on 17 pre-accident features."),
            ("6", "Dashboard", "Streamlit app with Folium map, Plotly charts, and live predictor."),
        ]
        for dot, title, desc in steps:
            st.markdown(f"""
            <div class="timeline-step">
                <div class="timeline-dot">{dot}</div>
                <div class="timeline-body">
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_stack:
        st.markdown('<div class="section-label">Technology</div><div class="section-title">Tech Stack</div>', unsafe_allow_html=True)
        stack = {
            "🧠 ML / AI": ["XGBoost", "scikit-learn (DBSCAN, LabelEncoder)", "Pandas, NumPy"],
            "🗺️ Visualisation": ["Plotly", "Folium + streamlit-folium"],
            "🖥️ App Layer": ["Streamlit", "Joblib (model serialisation)"],
            "📊 Data": ["cleaned_accidents.csv", "hotspots.csv", ".pkl encoders"],
        }
        for category, items in stack.items():
            st.markdown(f"""
            <div style='background:#0c1629;border:1px solid rgba(255,255,255,0.07);
                        border-radius:12px;padding:16px 20px;margin-bottom:12px;'>
                <div style='font-size:13px;font-weight:700;color:#f59e0b;margin-bottom:8px;'>{category}</div>
                {"".join(f'<span style="display:inline-block;background:rgba(255,255,255,0.05);border-radius:6px;padding:3px 10px;font-size:12px;color:#94a3b8;margin:3px 4px 3px 0;">{i}</span>' for i in items)}
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style='background:rgba(239,68,68,0.07);border:1px solid rgba(239,68,68,0.2);
                border-radius:12px;padding:20px 24px;max-width:750px;'>
        <div style='font-size:12px;font-weight:700;letter-spacing:2px;text-transform:uppercase;
                    color:#f87171;margin-bottom:8px;'>⚠️  Known Limitation</div>
        <p style='font-size:13px;color:#94a3b8;line-height:1.6;margin:0;'>
            The severity model currently includes <code style="color:#fcd34d;">casualties</code> and
            <code style="color:#fcd34d;">vehicles_involved</code> as training features — both are
            post-accident outcomes (data leakage). The Predictor fills these with dataset medians
            rather than asking the user. For a fully deployment-clean model, retrain without these
            two columns.
        </p>
    </div>
    """, unsafe_allow_html=True)