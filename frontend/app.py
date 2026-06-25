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

        # Feature contributions chart
        st.markdown("#### Risk factor contributions")
        contrib_df = pd.DataFrame({
            "Feature": list(contribs.keys()),
            "Contribution": list(contribs.values())
        }).sort_values("Contribution", ascending=True)
        contrib_df = contrib_df[contrib_df["Contribution"] != 0]

        if not contrib_df.empty:
            colors = ["#d03b3b" if v > 0 else "#0ca30c" for v in contrib_df["Contribution"]]
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
                xaxis_title="Risk contribution",
                font=dict(size=12, color="#52514e"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature contributions not available when using trained model (use SHAP for full explainability).")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.title("Prediction Dashboard")
    st.markdown("Historical overview of all predictions made via this system.")

    try:
        resp = requests.get(f"{API_URL}/history?limit=500", timeout=5)
        data = resp.json()
        history = data.get("history", [])
    except Exception:
        history = []

    if not history:
        st.info("No predictions yet. Run some predictions first.")
        st.stop()

    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # KPIs
    total = len(df)
    healthy = (df["prediction"] == "Healthy").sum()
    unhealthy = total - healthy
    avg_risk = df["risk_score"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Predictions", total)
    k2.metric("Healthy", healthy, delta=f"{healthy/total*100:.0f}%")
    k3.metric("At Risk", unhealthy, delta=f"{unhealthy/total*100:.0f}%", delta_color="inverse")
    k4.metric("Avg Risk Score", f"{avg_risk:.3f}")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Prediction distribution")
        fig_pie = px.pie(
            names=["Healthy", "At Risk"],
            values=[healthy, unhealthy],
            color_discrete_sequence=["#0ca30c", "#d03b3b"],
            hole=0.45,
        )
        fig_pie.update_layout(paper_bgcolor="white", margin=dict(t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown("#### Risk score distribution")
        fig_hist = px.histogram(
            df, x="risk_score", nbins=25,
            color_discrete_sequence=["#2a78d6"],
        )
        fig_hist.update_layout(
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis_title="Risk Score", yaxis_title="Count",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # BMI vs Risk scatter
    if "bmi" in df.columns:
        st.markdown("#### BMI vs. Risk Score")
        fig_sc = px.scatter(
            df, x="bmi", y="risk_score",
            color="prediction",
            color_discrete_map={"Healthy": "#0ca30c", "Unhealthy": "#d03b3b"},
            opacity=0.6,
        )
        fig_sc.update_layout(
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis_title="BMI", yaxis_title="Risk Score",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    # Recent predictions table
    st.markdown("#### Recent predictions")
    display_cols = ["timestamp", "age", "bmi", "blood_pressure", "glucose_level", "prediction", "risk_score", "confidence"]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available].tail(20).sort_values("timestamp", ascending=False), use_container_width=True)

    if st.button("🗑 Clear history", type="secondary"):
        requests.delete(f"{API_URL}/history")
        st.success("History cleared.")
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("About this Project")

    st.markdown("""
    ## NovaGen Health Risk Classifier

    A production-grade machine learning system built for **NovaGen Research Labs** to classify
    individuals as **healthy** or **at risk** based on 22 physiological and lifestyle features.

    ### Architecture
    | Layer | Technology | Role |
    |---|---|---|
    | **Model** | Random Forest (200 trees) | Core prediction engine |
    | **API** | FastAPI + Uvicorn | RESTful prediction endpoint |
    | **UI** | Streamlit | Interactive web interface |
    | **Containers** | Docker + Docker Compose | Deployment & isolation |

    ### Dataset
    - **9,800 unique participants** across multiple observational studies
    - **22 features**: physiological measurements, lifestyle factors, medical history
    - **Target**: Binary (Healthy / Unhealthy)

    ### Model Performance
    | Model | Accuracy | Recall |
    |---|---|---|
    | **Random Forest** ✅ | **93.7%** | **~95%** |
    | Gradient Boosting | ~91% | ~91% |
    | Voting Classifier | ~90% | ~90% |
    | Logistic Regression | ~82% | ~80% |
    | KNN (k=5) | ~79% | ~78% |

    > **Why recall?** In clinical screening, missing a high-risk patient (false negative)
    > is more dangerous than a false alarm. Recall was the primary optimization metric.

    ### API Endpoints
    | Method | Endpoint | Description |
    |---|---|---|
    | GET | `/` | API status |
    | GET | `/health` | Health check |
    | POST | `/predict` | Submit prediction |
    | GET | `/history` | Fetch history |
    | DELETE | `/history` | Clear history |

    ---
    Built by **Shivang Kathait** · shivangkathait@gmail.com
    """)
