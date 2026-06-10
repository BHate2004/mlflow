"""
Gurgaon Real Estate Price Predictor — Streamlit UI
Connects to FastAPI /predict endpoint (or falls back to local model).
"""

import streamlit as st
import requests
import json
import os
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gurgaon Property Price Estimator",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.stApp { background: #0d0f14; color: #e8e6e0; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

.hero {
    background: linear-gradient(135deg, #111318 0%, #1a1d26 60%, #0f1520 100%);
    border: 1px solid #2a2e3d;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(255,165,0,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    color: #f5a623;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1.15;
    color: #f0ece4;
    margin: 0 0 0.5rem 0;
}
.hero-title span { color: #f5a623; }
.hero-sub {
    font-size: 1rem;
    color: #8a8d99;
    max-width: 520px;
    line-height: 1.6;
}

.card {
    background: #111318;
    border: 1px solid #22262f;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.2rem;
}
.card-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    color: #f5a623;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.result-panel {
    background: linear-gradient(135deg, #151a10 0%, #0f1a08 100%);
    border: 1px solid #3a5220;
    border-radius: 14px;
    padding: 2.2rem 2.4rem;
    text-align: center;
}
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    color: #7abf4a;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.result-price {
    font-size: 3.6rem;
    font-weight: 700;
    color: #a8e063;
    line-height: 1;
    margin-bottom: 0.3rem;
    font-family: 'DM Mono', monospace;
}
.result-unit { font-size: 1rem; color: #5d7a40; margin-bottom: 1.2rem; }
.result-model {
    display: inline-block;
    background: rgba(90,180,40,0.12);
    border: 1px solid #3a5220;
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.75rem;
    color: #7abf4a;
    font-family: 'DM Mono', monospace;
}

.error-panel {
    background: #1a0f0f;
    border: 1px solid #5a2020;
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    color: #e07070;
    font-size: 0.9rem;
}

.metric-tile {
    background: #111318;
    border: 1px solid #22262f;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    text-align: center;
}
.metric-val {
    font-family: 'DM Mono', monospace;
    font-size: 1.5rem;
    font-weight: 500;
    color: #f5a623;
}
.metric-key { font-size: 0.75rem; color: #666a7a; margin-top: 0.2rem; }

div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stSlider"] label {
    font-size: 0.82rem !important;
    color: #9a9dab !important;
    font-weight: 500 !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: #181b22 !important;
    border: 1px solid #2a2e3d !important;
    border-radius: 8px !important;
    color: #e8e6e0 !important;
}
div[data-testid="stNumberInput"] input {
    background: #181b22 !important;
    border: 1px solid #2a2e3d !important;
    border-radius: 8px !important;
    color: #e8e6e0 !important;
}

div[data-testid="stButton"] > button {
    background: #f5a623 !important;
    color: #0d0f14 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.03em !important;
    transition: opacity 0.15s !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
div[data-testid="stButton"] > button:hover { opacity: 0.88 !important; }

hr { border-color: #22262f; }
</style>
""", unsafe_allow_html=True)


# ── Config & Data ──────────────────────────────────────────────────────────────

API_URL = os.getenv("API_URL", "http://localhost:8000")

SECTORS = sorted([
    "sector 1", "sector 2", "sector 4", "sector 5", "sector 6", "sector 7",
    "sector 9", "sector 10", "sector 11", "sector 12", "sector 13", "sector 14",
    "sector 15", "sector 16", "sector 17", "sector 18", "sector 19", "sector 20",
    "sector 21", "sector 22", "sector 23", "sector 24", "sector 25", "sector 26",
    "sector 27", "sector 28", "sector 29", "sector 30", "sector 31", "sector 32",
    "sector 33", "sector 34", "sector 35", "sector 36", "sector 37", "sector 38",
    "sector 39", "sector 40", "sector 41", "sector 42", "sector 43", "sector 44",
    "sector 45", "sector 46", "sector 47", "sector 48", "sector 49", "sector 50",
    "sector 51", "sector 52", "sector 53", "sector 54", "sector 55", "sector 56",
    "sector 57", "sector 58", "sector 59", "sector 60", "sector 61", "sector 62",
    "sector 63", "sector 64", "sector 65", "sector 66", "sector 67", "sector 68",
    "sector 69", "sector 70", "sector 71", "sector 72", "sector 73", "sector 74",
    "sector 75", "sector 76", "sector 77", "sector 78", "sector 79", "sector 80",
    "sector 81", "sector 82", "sector 83", "sector 84", "sector 85", "sector 86",
    "sector 87", "sector 88", "sector 89", "sector 90", "sector 91", "sector 92",
    "sector 93", "sector 95", "sector 99", "sector 102", "sector 104", "sector 105",
    "sector 106", "sector 108", "sector 109", "sector 110", "sector 111",
    "dwarka expressway", "golf course road", "golf course extension road",
    "mg road", "sohna road", "new colony", "palam vihar", "south city",
    "nirvana country", "vatika india next",
])

AGE_POSSESSION = [
    "New Property", "Relatively New", "Moderately Old", "Old Property", "Under Construction",
]

LUXURY_SCORES  = ["Low", "Medium", "High"]
BALCONY_OPTIONS = ["0", "1", "2", "3", "3+"]


# ── Prediction helpers ─────────────────────────────────────────────────────────

def predict_via_api(payload: dict) -> dict:
    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def predict_local(payload: dict) -> dict:
    import joblib, pathlib
    model_paths = ["models/best_model.joblib", "best_model.joblib"]
    for p in model_paths:
        if pathlib.Path(p).exists():
            import pandas as pd
            model = joblib.load(p)
            features = {
                "property_type":  payload["property_type"],
                "sector":         payload["sector"],
                "bedRoom":        payload["bedRoom"],
                "bathroom":       payload["bathroom"],
                "balcony":        payload["balcony"],
                "agePossession":  payload["agePossession"],
                "built_up_area":  payload["built_up_area"],
                "servant room":   payload["servant_room"],
                "store room":     payload["store_room"],
                "furnishing_type":payload["furnishing_type"],
                "luxury_score":   payload["luxury_score"],
            }
            pred_log = model.predict(pd.DataFrame([features]))[0]
            return {
                "predicted_price_cr": round(float(np.expm1(pred_log)), 2),
                "model_version": "RandomForest (local)",
            }
    raise FileNotFoundError("Model file not found — run training first.")


def get_prediction(payload: dict):
    try:
        return predict_via_api(payload), None
    except Exception as api_err:
        try:
            return predict_local(payload), f"API unavailable — used local model ({api_err})"
        except Exception as local_err:
            return None, f"Prediction failed.\n\nAPI: {api_err}\nLocal: {local_err}"


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">MLOps · Gurgaon Real Estate</div>
    <div class="hero-title">What's your property <span>worth?</span></div>
    <div class="hero-sub">
        RandomForest model trained on Gurgaon listings — fill in your property details
        and get an instant price estimate in Crore INR.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Model info strip ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-tile"><div class="metric-val">RandomForest</div><div class="metric-key">Best model</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-tile"><div class="metric-val">0.536</div><div class="metric-key">RMSE (Crore INR)</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-tile"><div class="metric-val">11</div><div class="metric-key">Input features</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-tile"><div class="metric-val">log1p</div><div class="metric-key">Target transform</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Form + Result ──────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<div class="card"><div class="card-label">01 · Location & Type</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        property_type = st.selectbox("Property type", ["flat", "house"])
    with col_b:
        sector = st.selectbox("Sector", SECTORS, index=SECTORS.index("sector 36") if "sector 36" in SECTORS else 0)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-label">02 · Size & Rooms</div>', unsafe_allow_html=True)
    col_c, col_d, col_e = st.columns(3)
    with col_c:
        built_up_area = st.number_input("Built-up area (sq ft)", min_value=100, max_value=20000, value=1200, step=50)
    with col_d:
        bedRoom = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    with col_e:
        bathroom = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    col_f, col_g, col_h = st.columns(3)
    with col_f:
        balcony = st.selectbox("Balconies", BALCONY_OPTIONS, index=2)
    with col_g:
        servant_room = st.selectbox("Servant room", [0, 1], format_func=lambda x: "Yes" if x else "No")
    with col_h:
        store_room = st.selectbox("Store room", [0, 1], format_func=lambda x: "Yes" if x else "No")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-label">03 · Condition & Furnishing</div>', unsafe_allow_html=True)
    col_i, col_j, col_k = st.columns(3)
    with col_i:
        agePossession = st.selectbox("Age / possession", AGE_POSSESSION)
    with col_j:
        furnishing_type = st.selectbox("Furnishing", ["unfurnished", "semiunfurnished", "furnished"])
    with col_k:
        luxury_score = st.selectbox("Luxury score", LUXURY_SCORES)
    st.markdown("</div>", unsafe_allow_html=True)

    predict_btn = st.button("Estimate Price →")

with right:
    st.markdown("<br>", unsafe_allow_html=True)

    if "result" not in st.session_state:
        st.session_state.result = None
        st.session_state.result_error = None
        st.session_state.result_warn = None

    if predict_btn:
        payload = {
            "property_type":  property_type,
            "sector":         sector,
            "bedRoom":        int(bedRoom),
            "bathroom":       int(bathroom),
            "balcony":        balcony,
            "agePossession":  agePossession,
            "built_up_area":  float(built_up_area),
            "servant_room":   int(servant_room),
            "store_room":     int(store_room),
            "furnishing_type":furnishing_type,
            "luxury_score":   luxury_score,
        }
        with st.spinner("Running model inference…"):
            result, err = get_prediction(payload)
        st.session_state.result = result
        st.session_state.result_error = err if result is None else None
        st.session_state.result_warn  = err if result is not None and err else None

    if st.session_state.result:
        r     = st.session_state.result
        price = r.get("predicted_price_cr", "—")
        model_ver = r.get("model_version", "RandomForest")

        st.markdown(f"""
        <div class="result-panel">
            <div class="result-label">Estimated Price</div>
            <div class="result-price">₹{price}</div>
            <div class="result-unit">Crore INR</div>
            <div class="result-model">{model_ver}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.result_warn:
            st.markdown(f'<div style="font-size:0.75rem;color:#888;margin-top:0.8rem;padding:0.6rem 1rem;background:#111318;border-radius:8px;border:1px solid #2a2e3d;">⚠ {st.session_state.result_warn}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card"><div class="card-label">Input summary</div>', unsafe_allow_html=True)
        for k, v in {
            "Type":        property_type.title(),
            "Sector":      sector.title(),
            "Area":        f"{built_up_area} sq ft",
            "Beds / Baths":f"{bedRoom} / {bathroom}",
            "Furnishing":  furnishing_type.title(),
            "Age":         agePossession,
            "Luxury":      luxury_score,
        }.items():
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:0.3rem 0;'
                f'border-bottom:1px solid #1e2230;font-size:0.82rem;">'
                f'<span style="color:#666a7a;">{k}</span>'
                f'<span style="color:#c8c5be;">{v}</span></div>',
                unsafe_allow_html=True
            )
        if isinstance(price, (int, float)) and built_up_area > 0:
            ppsf = round((price * 1e7) / built_up_area)
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:0.4rem 0;font-size:0.85rem;margin-top:0.4rem;">'
                f'<span style="color:#9a9dab;font-weight:600;">Price / sq ft</span>'
                f'<span style="color:#f5a623;font-family:\'DM Mono\',monospace;">₹{ppsf:,}</span></div>',
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state.result_error:
        st.markdown(f'<div class="error-panel">⚠ {st.session_state.result_error}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#111318;border:1px dashed #2a2e3d;border-radius:14px;
                    padding:3rem 2rem;text-align:center;color:#3d4155;">
            <div style="font-size:2.5rem;margin-bottom:0.8rem;">🏙️</div>
            <div style="font-size:0.9rem;line-height:1.6;">
                Fill in the property details<br>and click <strong style="color:#5a5f70;">Estimate Price →</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-size:0.75rem;color:#3d4155;padding-bottom:1rem;font-family:'DM Mono',monospace;">
    Gurgaon MLOps Pipeline · FastAPI + MLflow + Airflow · Model: RandomForest · Target: log1p(price)
</div>
""", unsafe_allow_html=True)