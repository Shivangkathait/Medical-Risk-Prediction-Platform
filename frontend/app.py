import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "https://medical-risk-prediction-platform.onrender.com")

st.set_page_config(
    page_title="NovaGen Health Risk Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f5f4f0; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e1e0d9; }
.metric-card {
    background: white; border-radius: 12px; padding: 20px;
    border: 0.5px solid #e1e0d9; margin-bottom: 12px;
}
.result-healthy {
    background: #eaf3de; border-radius: 12px; padding: 24px;
    border: 1px solid #0ca30c; margin-top: 16px;
}
.result-unhealthy {
    background: #fcebeb; border-radius: 12px; padding: 24px;
    border: 1px solid #d03b3b; margin-top: 16px;
}
.result-title-healthy { color: #0ca30c; font-size: 28px; font-weight: 700; margin: 0; }
.result-title-unhealthy { color: #d03b3b; font-size: 28px; font-weight: 700; margin: 0; }
.result-sub { color: #52514e; font-size: 15px; margin-top: 4px; }
.section-header {
    font-size: 13px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #898781; margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 NovaGen")
    st.markdown("**Health Risk Classifier**")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔬 Predict", "📊 Dashboard", "ℹ️ About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#898781'>Powered by Random Forest · 93.7% accuracy</small>",
        unsafe_allow_html=True
    )
    # API status
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            st.success("API: Online", icon="✅")
        else:
            st.error("API: Error")
    except Exception:
        st.error("API: Offline")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔬 Predict":
    st.title("Health Risk Assessment")
    st.markdown("Fill in the patient's health indicators to get an instant risk prediction.")

    with st.form("prediction_form"):
        # ── Physiological Measurements ────────────────────────────────────────
        st.markdown('<p class="section-header">Physiological measurements</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=35)
            blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=60, max_value=250, value=120)
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=72)
        with col2:
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0, step=0.1)
            cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=185)
            glucose_level = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=300, value=90)
        with col3:
            blood_group = st.selectbox("Blood Group", ["A", "B", "AB", "O"])
            medical_history = st.selectbox("Prior Medical Conditions", ["No", "Yes"])
            allergies = st.selectbox("Known Allergies", ["No", "Yes"])

        st.markdown("---")
        # ── Lifestyle Factors ─────────────────────────────────────────────────
        st.markdown('<p class="section-header">Lifestyle factors</p>', unsafe_allow_html=True)
        col4, col5, col6 = st.columns(3)
        with col4:
            sleep_hours = st.slider("Sleep Hours / day", 1.0, 12.0, 7.0, 0.5)
            exercise_hours = st.slider("Exercise Hours / day", 0.0, 5.0, 0.5, 0.1)
        with col5:
            water_intake = st.slider("Water Intake (litres)", 0.5, 6.0, 2.5, 0.5)
            stress_level = st.slider("Stress Level (1–10)", 1, 10, 4)
        with col6:
            physical_activity = st.slider("Physical Activity (1–10)", 1, 10, 5)
            mental_health = st.slider("Mental Health Score (1–10)", 1, 10, 7)

        st.markdown("---")
        # ── Habits & Diet ─────────────────────────────────────────────────────
        st.markdown('<p class="section-header">Habits & diet</p>', unsafe_allow_html=True)
        col7, col8, col9 = st.columns(3)
        with col7:
            smoking = st.selectbox("Smoking", ["Non-smoker", "Smoker"])
            alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
        with col8:
            diet_type = st.selectbox("Diet Type", ["Non-vegetarian", "Vegetarian", "Vegan"])
        with col9:
            st.empty()

        submitted = st.form_submit_button("🔬 Run Prediction", use_container_width=True, type="primary")

    # ── Handle Submission ─────────────────────────────────────────────────────
    if submitted:
        payload = {
            "age": age,
            "bmi": bmi,
            "blood_pressure": blood_pressure,
            "cholesterol": cholesterol,
            "glucose_level": glucose_level,
            "heart_rate": heart_rate,
            "sleep_hours": sleep_hours,
            "exercise_hours": exercise_hours,
            "water_intake": water_intake,
            "stress_level": stress_level,
            "smoking": 1 if smoking == "Smoker" else 0,
            "alcohol": 1 if alcohol == "Yes" else 0,
            "diet": {"Non-vegetarian": 0, "Vegetarian": 1, "Vegan": 2}[diet_type],
            "mental_health": mental_health,
            "physical_activity": physical_activity,
            "medical_history": 1 if medical_history == "Yes" else 0,
            "allergies": 1 if allergies == "Yes" else 0,
            "diet_type_vegan": 1 if diet_type == "Vegan" else 0,
            "diet_type_vegetarian": 1 if diet_type == "Vegetarian" else 0,
            "blood_group_ab": 1 if blood_group == "AB" else 0,
            "blood_group_b": 1 if blood_group == "B" else 0,
            "blood_group_o": 1 if blood_group == "O" else 0,
        }

        with st.spinner("Analyzing health profile..."):
            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                resp.raise_for_status()
                result = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API. Make sure the backend is running.")
                st.stop()
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        pred = result["prediction"]
        conf = result["confidence"]
        risk = result["risk_score"]
        rec  = result["recommendation"]
        contribs = result["feature_contributions"]

        # Result card
        css_class = "result-healthy" if pred == "Healthy" else "result-unhealthy"
        title_class = "result-title-healthy" if pred == "Healthy" else "result-title-unhealthy"
        icon = "✅" if pred == "Healthy" else "⚠️"
        st.markdown(f"""
        <div class="{css_class}">
            <p class="{title_class}">{icon} {pred}</p>
            <p class="result-sub">{rec}</p>
        </div>
        """, unsafe_allow_html=True)

        # Metrics row
        st.markdown("")
        m1, m2, m3 = st.columns(3)
        m1.metric("Confidence", f"{conf * 100:.1f}%")
        m2.metric("Risk Score", f"{risk:.3f}")
        m3.metric("Assessment Time", datetime.utcnow().strftime("%H:%M:%S UTC"))

        # Feature contributions (shown only if available)
        if contribs:
            st.markdown("#### Risk Factor Contributions")

            contrib_df = pd.DataFrame({
                "Feature": list(contribs.keys()),
                "Contribution": list(contribs.values())
            }).sort_values("Contribution", ascending=True)

            contrib_df = contrib_df[contrib_df["Contribution"] != 0]

            if not contrib_df.empty:
                colors = [
                    "#d03b3b" if v > 0 else "#0ca30c"
                    for v in contrib_df["Contribution"]
                ]

                fig = go.Figure(go.Bar(
                    x=contrib_df["Contribution"],
                    y=contrib_df["Feature"],
                    orientation="h",
                    marker_color=colors,
                ))

                fig.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    height=300,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title="Contribution",
                )

                st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("About this Project")

    st.markdown("""
    ## 🧬 NovaGen Health Risk Classifier

    A production-ready Machine Learning application for predicting patient health risk using a
    **Random Forest Classifier** with a **FastAPI backend**, **Streamlit frontend**, and **Docker-based deployment**.

    ### 🏗 Architecture

    | Layer | Technology | Purpose |
    |-------|------------|----------|
    | **Machine Learning** | Random Forest | Health risk prediction |
    | **Backend API** | FastAPI + Uvicorn | REST API |
    | **Frontend** | Streamlit | Interactive Dashboard |
    | **Deployment** | Docker, Render, Streamlit Cloud | Production deployment |

    ### 📊 Dataset

    - **9,800+ health records**
    - **22 clinical & lifestyle features**
    - Binary Classification (**Healthy / At Risk**)

    ### 🤖 Model Performance

    | Model | Accuracy | Recall |
    |------|---------:|-------:|
    | ✅ Random Forest | **93.7%** | **95%** |
    | Gradient Boosting | 91% | 91% |
    | Voting Classifier | 90% | 90% |
    | Logistic Regression | 82% | 80% |
    | KNN | 79% | 78% |

    > Higher Recall was prioritized because identifying high-risk patients is more important than minimizing false alarms.

    ### 🔌 API Endpoints

    | Method | Endpoint | Description |
    |--------|----------|-------------|
    | GET | `/health` | API Health Check |
    | POST | `/predict` | Predict Health Risk |
    | GET | `/history` | Prediction History |
    | DELETE | `/history` | Clear History |

    <br><hr>

    <div style="text-align:center; color:#6b7280;">
    © 2026 <b>Shivang Kathait</b><br>
    📧 shivangkathait@gmail.com
    </div>
    """, unsafe_allow_html=True)
