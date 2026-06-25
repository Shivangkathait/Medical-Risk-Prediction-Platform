# 🧬 NovaGen Health Risk Classifier

> Production-grade ML system — Binary health classification using Random Forest  
> **FastAPI · Streamlit · Docker · scikit-learn**

---

## Architecture

```
novagen/
├── backend/
│   ├── app/
│   │   └── main.py          # FastAPI application
│   ├── train_model.py        # Model training script
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py               # Streamlit UI (3 pages)
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

```
┌─────────────────────┐        ┌──────────────────────┐
│   Streamlit UI      │  HTTP  │    FastAPI Backend    │
│   :8501             │───────▶│    :8000              │
│                     │        │                       │
│  • Predict form     │        │  POST /predict        │
│  • Dashboard        │        │  GET  /history        │
│  • About            │        │  GET  /health         │
└─────────────────────┘        └──────────┬───────────┘
                                           │
                               ┌──────────▼───────────┐
                               │  Random Forest Model  │
                               │  StandardScaler       │
                               │  (trained on build)   │
                               └──────────────────────┘
```

---

## Quick Start

### Prerequisites
- [Docker](https://www.docker.com/get-started) + Docker Compose v2
- 4 GB RAM recommended

### 1. Clone / download the project
```bash
git clone <your-repo-url>
cd novagen
```

### 2. Build and run
```bash
docker compose up --build
```

> First build trains the Random Forest model (~30–60 seconds).  
> Subsequent starts are instant.

### 3. Open the app
| Service | URL |
|---|---|
| **Streamlit UI** | http://localhost:8501 |
| **FastAPI docs** | http://localhost:8000/docs |
| **API (raw)** | http://localhost:8000 |

---

## Usage

### Predict page
Fill in the patient health form → click **Run Prediction** → see:
- Healthy / At Risk verdict with confidence %
- Risk score (0–1)
- Feature contribution bar chart
- Clinical recommendation

### Dashboard page
- Live stats: total predictions, healthy/at-risk split, avg risk score
- Pie chart, histogram, BMI vs risk scatter
- Scrollable recent predictions table

### About page
- Model comparison table
- Full API endpoint reference
- Architecture overview

---

## API Reference

**Base URL:** `http://localhost:8000`

### POST `/predict`
Submit a patient record and get a prediction.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "bmi": 28.5,
    "blood_pressure": 135,
    "cholesterol": 210,
    "glucose_level": 105,
    "heart_rate": 76,
    "sleep_hours": 6.5,
    "exercise_hours": 0.5,
    "water_intake": 2.0,
    "stress_level": 6,
    "smoking": 0,
    "alcohol": 1,
    "diet": 1,
    "mental_health": 7,
    "physical_activity": 5,
    "medical_history": 0,
    "allergies": 0,
    "diet_type_vegan": 0,
    "diet_type_vegetarian": 1,
    "blood_group_ab": 0,
    "blood_group_b": 0,
    "blood_group_o": 1
  }'
```

**Response:**
```json
{
  "prediction": "Healthy",
  "confidence": 0.823,
  "risk_score": 0.177,
  "recommendation": "Good health profile. Continue regular exercise and balanced diet.",
  "feature_contributions": { "BMI": 0.07, "Smoking": 0.0, ... },
  "timestamp": "2025-01-01T12:00:00"
}
```

### GET `/history?limit=100`
Returns the last N predictions.

### DELETE `/history`
Clears all stored predictions.

### GET `/health`
```json
{ "status": "ok", "model_loaded": true, "timestamp": "..." }
```

---

## Model Details

| Attribute | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Trees | 200 (`n_estimators=200`) |
| Dataset size | 9,800 records |
| Features | 22 (physiological + lifestyle) |
| Train/Test split | 80% / 20%, stratified |
| Preprocessing | StandardScaler |
| Accuracy | ~93.7% |
| Recall | ~95% |
| Primary metric | **Recall** (false negatives more costly) |

### Feature set
`age` · `bmi` · `blood_pressure` · `cholesterol` · `glucose_level` · `heart_rate` · `sleep_hours` · `exercise_hours` · `water_intake` · `stress_level` · `smoking` · `alcohol` · `diet` · `mental_health` · `physical_activity` · `medical_history` · `allergies` · `diet_type_vegan` · `diet_type_vegetarian` · `blood_group_ab` · `blood_group_b` · `blood_group_o`

---

## Development

### Run without Docker

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python train_model.py        # generates models/
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
pip install -r requirements.txt
API_URL=http://localhost:8000 streamlit run app.py
```

### Rebuild containers after code changes
```bash
docker compose up --build
```

### Stop and clean up
```bash
docker compose down          # stop containers
docker compose down -v       # stop + delete volume (clears history)
```

---

## Deployment (Production)

### Railway / Render
1. Push to GitHub
2. Connect repo in Railway / Render
3. Set `API_URL` env var on the frontend service to your backend's public URL
4. Deploy both services

### AWS / GCP
- Package with `docker compose` and push images to ECR / Artifact Registry
- Run on ECS Fargate / Cloud Run
- Use managed Postgres instead of JSON file for history persistence

---

## Tech Stack
`Python 3.11` · `FastAPI` · `Uvicorn` · `scikit-learn` · `Streamlit` · `Plotly` · `Docker` · `Docker Compose` · `Pydantic v2`

---

## Author
**Shivang Kathait** · shivangkathait@gmail.com
