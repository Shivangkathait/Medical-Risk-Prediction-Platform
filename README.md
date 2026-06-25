# 🧬 Medical Risk Prediction Platform

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow)

> A production-ready Machine Learning application for predicting patient health risk using a **Random Forest Classifier**, featuring an interactive **Streamlit** dashboard, **FastAPI** backend, and Dockerized deployment.

**FastAPI • Streamlit • Scikit-learn • Docker • Docker Compose • Render**

---

# 🌐 Live Demo

| Service              | Link                                                       |
| -------------------- | ---------------------------------------------------------- |
| 🚀 Web Application   | https://medical-risk-prediction.streamlit.app/             |
| ⚡ Backend API        | https://medical-risk-prediction-platform.onrender.com      |
| 📖 API Documentation | https://medical-risk-prediction-platform.onrender.com/docs |

---

# ✨ Features

* 🩺 Real-time Health Risk Prediction
* 🤖 Random Forest Machine Learning Model
* 📊 Interactive Dashboard with Charts
* ⚡ FastAPI REST API
* 📜 Prediction History Tracking
* 🐳 Docker & Docker Compose Support
* ☁️ Cloud Deployment (Render + Streamlit Community Cloud)

---

# 🏗️ Architecture

```text
Streamlit UI
      │
      ▼
 FastAPI REST API
      │
      ▼
Random Forest Model
      │
      ▼
Prediction History
```

---

# 🛠️ Tech Stack

* Python
* FastAPI
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Plotly
* Docker
* Docker Compose
* Uvicorn

---

# 🚀 Run Locally

```bash
git clone https://github.com/Shivangkathait/Medical-Risk-Prediction-Platform.git

cd Medical-Risk-Prediction-Platform

docker compose up --build
```

### Local URLs (Docker)

* **Frontend:** http://localhost:8501
* **Backend API:** http://localhost:8000
* **API Documentation:** http://localhost:8000/docs

> **Note:** These localhost URLs are available only when running the project locally using Docker.

---

# 🔌 API Endpoints

| Method | Endpoint   | Description                 |
| ------ | ---------- | --------------------------- |
| GET    | `/health`  | Health Check                |
| POST   | `/predict` | Predict Health Risk         |
| GET    | `/history` | Retrieve Prediction History |
| DELETE | `/history` | Clear Prediction History    |

---

# 📂 Project Structure

```text
backend/
├── app/
├── train_model.py
└── Dockerfile

frontend/
├── app.py
└── Dockerfile

docker-compose.yml
README.md
```

---

# 📄 License

This project is available under the **MIT License**.

---

# 👨‍💻 Author

**Shivang Kathait**

📧 **[shivangkathait@gmail.com](mailto:shivangkathait@gmail.com)**

🔗 GitHub: https://github.com/Shivangkathait

---

⭐ **If you found this project useful, consider giving it a Star on GitHub!**
