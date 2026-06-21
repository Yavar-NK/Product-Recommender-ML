# End-to-End Production-Ready Product Recommender System

A modular, production-grade Machine Learning pipeline that leverages NLP embeddings to build an intelligent recommendation engine. This project transitions the initial model into an enterprise-level MLOps architecture, featuring structural code separation, robust experiment tracking, and complete containerization capabilities.

## 🚀 Key Features & Architecture
* **Modular Codebase:** Clean separation of concerns with dedicated modules for data loading, preprocessing, model training, and API deployment under the `src/` directory.
* **Production-Grade API:** Powered by **FastAPI** and served via **Uvicorn** for high-performance, asynchronous, and scalable model serving.
* **Experiment Tracking & MLOps:** Integrated with **MLflow** to seamlessly log hyperparameters, pipeline metrics, and model artifacts (`mlruns/`).
* **Containerization:** Out-of-the-box **Dockerfile** configured with optimized multi-layer caching (`python:3.10-slim`) to guarantee cross-environment reproducibility.

---

## 📂 Project Structure
```text
├── src/
│   ├── data_loader.py    # Robust data ingestion (automated Kaggle API downloads)
│   ├── preprocess.py    # Feature engineering and text embedding pipelines
│   ├── train.py         # Model training, validation, and MLflow logging
│   └── app.py           # Production FastAPI web service
├── models/              # Pre-trained models and saved weights
├── mlruns/              # MLflow localized tracking repository
├── Dockerfile           # Optimized production Docker deployment script
├── .gitignore           # Git exclusions for models and cache layers
└── requirements.txt     # Locked project dependencies
```
## How to Run 🛠️

### 1. Local Serving (FastAPI)
To launch the API server locally without containerization, run the following command from the root directory:

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000

```
### 2. Production Deployment (Docker)
The codebase includes a fully-optimized Docker environment. To build and run the containerized application:

```bash
docker build -t product-recommender:v1 .
docker run -p 8000:8000 product-recommender:v1
