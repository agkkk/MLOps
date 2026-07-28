# ABC Retail MLOps Architecture

An end-to-end Machine Learning Operations pipeline built to manage, deploy, and monitor retail models. This project is fully containerized and features a decoupled architecture using FastAPI, Streamlit, and MLflow.

## 🏗️ Architecture Overview
*   **Frontend:** Streamlit Dashboard for interactive data and model visualization.
*   **Backend Engine:** FastAPI server handling API requests and model serving.
*   **Model Tracking:** MLflow integration for logging metrics, parameters, and model artifacts.
*   **Automation:** Python `schedule` library for automated periodic tasks.
*   **Containerization:** Docker & Docker Compose for isolated, reproducible deployments.

## 🚀 How to Run the Project (Docker)
This entire architecture has been packaged into a single public Docker image. You do not need to clone this repository or install Python to run it.

Ensure Docker Desktop is installed and running, then execute the following command in your terminal:

`docker run -p 8000:8000 -p 8501:8501 ananthkowlagi/abc-retail-mlops:latest`

## 🌐 Accessing the Services
Once the container boots up, you can access the isolated services in your local browser:
*   **Streamlit Dashboard:** http://localhost:8501
*   **FastAPI Backend:** http://localhost:8000
