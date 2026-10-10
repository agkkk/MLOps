# ABC Retail Group — Intelligent Dashboard Generator (MLOps Project)

> **Workshop:** MLOps (2-Credit) | **Semester Gap:** 4th–5th Sem, B.Tech CSE
> **Data Source:** Global Superstore Dataset (Real-World, Public)
> **Tech Stack:** Python, Scikit-learn, FastAPI, Streamlit, MLflow, SQLite, Docker, Kubernetes

---

## Table of Contents
1. [Problem Statement (Explained Simply)](#1-problem-statement-explained-simply)
2. [What We Built (One-Line Summary)](#2-what-we-built)
3. [How to Run the Project](#3-how-to-run-the-project)
4. [Step-by-Step: What the Project Does](#4-step-by-step-what-the-project-does)
5. [Dashboard Tabs — What You See on Screen](#5-dashboard-tabs--what-you-see-on-screen)
6. [Models Trained & Results](#6-models-trained--results)
7. [Architecture Overview](#7-architecture-overview)
8. [Project File Structure](#8-project-file-structure)
9. [Login Credentials](#9-login-credentials)
10. [Key Talking Points for Presentation](#10-key-talking-points-for-presentation)

---

## 1. Problem Statement (Explained Simply)

### The Official Problem Statement
> *"ABC Retail Group operates across multiple cities and collects operational data from sales, inventory, finance, customer support, and logistics. Management currently relies on manually prepared reports. Design and develop an Intelligent Dashboard Generator that automatically analyzes business datasets, generates interactive dashboards, and detects anomalies using Machine Learning techniques."*

### What This Actually Means (In Plain English)
Imagine a big retail company (like Reliance Retail or DMart) that has stores in many cities. Every single day, their different departments generate tons of data:
- **Sales** department tracks: what items sold, how much revenue, how many customers walked in
- **Inventory** department tracks: how much stock is left, when to reorder
- **Finance** department tracks: operating costs, marketing spend, profits
- **Customer Support** tracks: how many complaints came in, how fast they were resolved
- **Logistics** tracks: how many shipments went out, how many were delayed

Right now, some manager has to manually open Excel sheets, make charts, and try to figure out what's going wrong. That's slow and error-prone.

**Our job:** Build a system that does ALL of this automatically:
1. Takes in raw business data
2. Cleans it (because real data is always messy — missing values, wrong formats, outliers)
3. Runs Machine Learning models on it to find patterns humans would miss
4. Shows everything on a nice interactive dashboard with charts and graphs
5. Specifically detects **anomalies** — unusual days where something went seriously wrong (e.g., half the shipments were delayed AND customer complaints spiked — that's a red flag)

---

## 2. What We Built

**An end-to-end MLOps pipeline** that takes real-world retail data, preprocesses it through 8 cleaning steps, trains and compares 15+ ML models across 3 business tasks, and displays everything on an authenticated interactive dashboard with a FastAPI backend for real-time predictions.

---

## 3. How to Run the Project

### Quick Start (Local — No Docker Needed)
```bash
# Step 1: Activate the Python virtual environment
source venv/bin/activate

# Step 2: Run the full pipeline (preprocessing → training → dashboard)
./run.sh
```

Once it boots up, open your browser:
- **Dashboard:** http://localhost:8501
- **FastAPI (API docs):** http://localhost:8000/docs

### What `run.sh` Does Internally
1. Runs `data_preprocessing.py` — loads raw CSV, cleans it, creates 5 department tables
2. Runs `train_models.py` — trains 15+ models, compares them, saves the best ones
3. Starts MLflow tracking server (background)
4. Starts FastAPI server on port 8000 (background)
5. Starts Streamlit dashboard on port 8501 (foreground)

### Login Credentials (See Section 9)

---

## 4. Step-by-Step: What the Project Does

### PHASE 1: Data Acquisition
- **What:** We use the **Global Superstore Dataset**, a real-world public retail dataset
- **Where from:** Kaggle / Public repositories (it's a widely-used dataset in analytics)
- **What's in it:** 1,000 retail transactions from across 147 countries, with 24 columns including: Order Date, Ship Date, Customer Name, Category (Technology/Furniture/Office Supplies), Sales amount, Quantity, Discount, Profit, Shipping Cost, etc.
- **File:** `src/data/raw_superstore.csv` (246 KB)

### PHASE 2: Data Preprocessing (`src/data_preprocessing.py`)
This is where we clean the raw, messy data. There are **8 steps**:

| Step | What It Does | Example |
|------|-------------|---------|
| 1. Load Raw Data | Read the CSV file with proper encoding | Loaded 1000 rows × 24 columns |
| 2. Remove Duplicates | Check for and remove repeated rows | 0 duplicates found in this dataset |
| 3. Handle Missing Values | Fill in blanks intelligently | 806 Postal Codes were blank → filled with "Unknown"; numeric blanks → filled with median |
| 4. Fix Data Types | Convert text dates to actual date objects, ensure numbers are numeric | "11/11/2014" → proper datetime |
| 5. Detect Outliers (IQR) | Flag unusually high/low values using the Interquartile Range method | Sales: 21 outliers (2.1%), Profit: 67 outliers (6.7%) |
| 6. Feature Engineering | Create NEW useful columns from existing data | `shipping_delay_days` = Ship Date − Order Date; `profit_margin` = Profit ÷ Sales; `revenue_per_unit` = Sales ÷ Quantity |
| 7. Transform to Departments | Split cleaned data into 5 business department tables | Sales (798 rows), Inventory (798), Finance (599), Support (599), Logistics (599) |
| 8. Save | Export to CSV files + SQLite database | 5 CSV files + 1 .db file |

**Input:** 1 raw CSV (1000 rows × 24 columns)
**Output:** 5 clean departmental CSVs + 1 SQLite database + a JSON report showing before/after stats

### PHASE 3: Model Training & Comparison (`src/ml/train_models.py`)
For each business task, we don't just train ONE model — we train MULTIPLE models and compare them, then pick the best one. This is what the guide specifically asked for.

#### Task 1: Revenue Forecasting
- **Goal:** Given today's footfall, marketing spend, and operating cost → predict today's revenue
- **Input features:** `footfall`, `marketing_spend`, `operating_cost`
- **Output:** Predicted revenue in dollars
- **Models trained (6):**

| Model | MAE ($) | RMSE ($) | R² Score | Verdict |
|-------|---------|----------|----------|---------|
| Linear Regression | 970.22 | 1303.11 | 0.6217 | Decent baseline |
| Ridge Regression | 970.29 | 1303.11 | 0.6217 | Same as Linear (expected) |
| Decision Tree | 443.88 | 677.79 | 0.8977 | Good but overfits |
| **Random Forest** | **402.49** | **568.08** | **0.9281** | **🏆 BEST — selected** |
| Gradient Boosting | 731.25 | 954.01 | 0.7973 | Good but not best |
| SVR (RBF Kernel) | 908.61 | 1370.17 | 0.5818 | Worst performer |

**Why Random Forest won:** Lowest error (MAE: $402) and highest accuracy (R²: 0.93 = 93% of variance explained).

#### Task 2: Anomaly Detection
- **Goal:** Detect days where operations went abnormally wrong (e.g., mass shipment delays + customer complaint spikes happening together)
- **Input features:** `delayed_shipments`, `tickets_opened`
- **Output:** "Is this day an anomaly? Yes/No"
- **Models trained (3):**

| Model | Anomalies Detected | Anomaly Rate |
|-------|-------------------|--------------|
| **Isolation Forest** | **30** | **5.01%** |
| Local Outlier Factor | 29 | 4.84% |
| One-Class SVM | 82 | 13.69% |

**Why Isolation Forest was selected:** Industry standard for unsupervised anomaly detection. One-Class SVM was too aggressive (flagged 14% as anomalies, which is unrealistic).

#### Task 3: Inventory Clustering
- **Goal:** Segment/group inventory days into business categories (good days vs bad days vs average days)
- **Input features:** `stock_level`, `quantity`, `discount`
- **Output:** A cluster label (e.g., "High Efficiency", "Average Day", "Problematic Day")
- **Models trained (6):**

| Model | Clusters | Silhouette Score |
|-------|----------|-----------------|
| K-Means (k=2) | 2 | 0.4448 |
| K-Means (k=3) | 3 | 0.4706 |
| K-Means (k=4) | 4 | 0.3784 |
| K-Means (k=5) | 5 | 0.3459 |
| Agglomerative (k=3) | 3 | 0.3648 |
| **DBSCAN** | **2** | **0.6442** |

**Why DBSCAN won:** Highest silhouette score (0.64). Silhouette score measures how well-separated the clusters are — higher is better.

#### Additionally: Linear Regression Trendline
- A simple Linear Regression model is also trained to show the mathematical relationship between footfall and revenue as a visual trendline on the dashboard.

### PHASE 4: Model Deployment
- Best models are saved as `.pkl` files (pickle/serialized Python objects)
- **FastAPI** server loads these `.pkl` files into memory on startup
- Exposes 4 REST API endpoints that anyone (or any application) can call:
  - `POST /predict/anomaly` — send shipment + ticket data, get "is anomaly?" back
  - `POST /predict/revenue` — send footfall + spend data, get predicted revenue back
  - `POST /predict/cluster` — send inventory data, get business category back
  - `POST /predict/trendline` — send footfall, get LR-based revenue prediction back
  - `GET /health` — check if API is alive

### PHASE 5: Dashboard & Visualization
- **Streamlit** dashboard with **authentication** (login required)
- 8 tabs showing different aspects of the pipeline
- Interactive Plotly charts (hoverable, zoomable)
- Live API testing from the dashboard itself
- CRUD operations (insert/delete records in the database and retrain models)

### PHASE 6: MLOps Infrastructure
- **MLflow** tracks experiments (parameters, metrics, model artifacts) for the revenue model
- **Docker** + **Kubernetes** configs provided for containerized deployment
- **Git** for version control
- **Automated scheduler** available for periodic retraining cycles

---

## 5. Dashboard Tabs — What You See on Screen

### Tab 1: 🔄 Pipeline Overview
- **What it shows:** The entire MLOps pipeline broken down into 7 phases with descriptions
- **Purpose:** Gives a bird's-eye view of the architecture — from data acquisition to automation
- **Also shows:** The 4 architecture components (Streamlit, FastAPI, MLflow, SQLite) as metric cards

### Tab 2: 🧹 Data Preprocessing
- **What it shows:**
  - **Before vs After** metrics: how many rows, columns, missing values before and after cleaning
  - The **8 preprocessing steps** that were executed, listed out
  - **Outlier detection chart** — a bar chart showing how many outliers were found per column
  - **Raw data sample** — what the messy data looked like BEFORE cleaning
  - **Cleaned departmental data** — sub-tabs showing the final clean tables (Sales, Inventory, Finance, Support, Logistics)
  - **Missing values summary** — which columns had nulls and how many
- **Purpose:** Directly addresses the guide's feedback about showing data cleaning and "how data looked before processing and how it looks after"

### Tab 3: 📈 Model Comparison
- **What it shows:**
  - For **Revenue Forecasting**: a metrics table (MAE, RMSE, R², CV R²) for all 6 models + two bar charts (R² comparison and MAE comparison) with the best model highlighted in green
  - For **Anomaly Detection**: a table + bar chart showing anomalies detected by each of the 3 models
  - For **Inventory Clustering**: a table + bar chart comparing Silhouette Scores of all 6 clustering approaches
- **Purpose:** Directly addresses the guide's feedback about "why only 3-4 models? compare many models"

### Tab 4: 🚨 Anomaly Detection
- **What it shows:** A scatter plot with delayed shipments on X-axis and support tickets on Y-axis. Normal days are grey dots, anomaly days are RED dots. Also shows total days analyzed, failures detected, and anomaly rate as metric cards.
- **Purpose:** Visually shows which days had operational failures

### Tab 5: 💰 Revenue Forecast
- **What it shows:** A line chart with two lines — blue (actual revenue) vs red dashed (predicted revenue by the best model). Shows MAE, R² score, and data points as metric cards.
- **Purpose:** Shows how well our model predicts revenue

### Tab 6: 📦 Inventory Clusters
- **What it shows:** A pie chart showing the distribution of business categories + a scatter plot (stock level vs quantity) colored by cluster
- **Purpose:** Shows inventory segmentation

### Tab 7: 📉 LR Trendline
- **What it shows:** Scatter plot of all data points (footfall vs revenue) with a linear regression line drawn through them
- **Purpose:** Shows the direct mathematical impact of footfall on revenue

### Tab 8: ⚡ Live API & CRUD
- **What it shows:**
  - A form where you can enter footfall, marketing spend, and operating cost → click "Predict" → it calls the FastAPI backend in real-time and shows the predicted revenue
  - A CRUD section where you can insert or delete sales records in the database and trigger a full retrain of all models
- **Purpose:** Demonstrates real-time model serving and database management

---

## 6. Models Trained & Results

### Summary Table (All 15 Models)

| # | Task | Model | Key Metric | Score | Selected? |
|---|------|-------|------------|-------|-----------|
| 1 | Revenue | Linear Regression | R² | 0.6217 | ❌ |
| 2 | Revenue | Ridge Regression | R² | 0.6217 | ❌ |
| 3 | Revenue | Decision Tree | R² | 0.8977 | ❌ |
| 4 | Revenue | **Random Forest** | **R²** | **0.9281** | ✅ Best |
| 5 | Revenue | Gradient Boosting | R² | 0.7973 | ❌ |
| 6 | Revenue | SVR (RBF) | R² | 0.5818 | ❌ |
| 7 | Anomaly | **Isolation Forest** | Anomaly Rate | **5.01%** | ✅ Selected |
| 8 | Anomaly | Local Outlier Factor | Anomaly Rate | 4.84% | ❌ |
| 9 | Anomaly | One-Class SVM | Anomaly Rate | 13.69% | ❌ |
| 10 | Clustering | K-Means (k=2) | Silhouette | 0.4448 | ❌ |
| 11 | Clustering | K-Means (k=3) | Silhouette | 0.4706 | ❌ |
| 12 | Clustering | K-Means (k=4) | Silhouette | 0.3784 | ❌ |
| 13 | Clustering | K-Means (k=5) | Silhouette | 0.3459 | ❌ |
| 14 | Clustering | Agglomerative (k=3) | Silhouette | 0.3648 | ❌ |
| 15 | Clustering | **DBSCAN** | **Silhouette** | **0.6442** | ✅ Best |

---

## 7. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ABC RETAIL MLOps PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  Raw Data     │───▶│ Preprocessing │───▶│ 5 Department     │  │
│  │  (Superstore) │    │ (8 Steps)     │    │ Tables (CSV+DB)  │  │
│  └──────────────┘    └──────────────┘    └───────┬──────────┘  │
│                                                   │              │
│                                          ┌────────▼─────────┐  │
│                                          │  ML Training      │  │
│                                          │  (15+ Models)     │  │
│                                          │  + Comparison     │  │
│                                          └────────┬─────────┘  │
│                                                   │              │
│         ┌─────────────────────────────────────────┤              │
│         │                                         │              │
│  ┌──────▼──────┐  ┌──────────────┐  ┌────────────▼───────────┐ │
│  │  FastAPI     │  │   MLflow      │  │  Streamlit Dashboard   │ │
│  │  (REST API)  │  │  (Tracking)   │  │  (8 Tabs + Auth)      │ │
│  │  Port 8000   │  │  Port 5000    │  │  Port 8501             │ │
│  └─────────────┘  └──────────────┘  └────────────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  DevOps: Docker │ Kubernetes │ Git │ Automated Scheduler    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Frontend | Streamlit | Interactive dashboard with 8 tabs |
| Backend API | FastAPI | REST endpoints for real-time model inference |
| ML Framework | Scikit-learn | Training all 15+ models |
| Experiment Tracking | MLflow | Logging parameters, metrics, and model artifacts |
| Database | SQLite | Storing departmental data |
| Data Processing | Pandas, NumPy | Cleaning, transformation, feature engineering |
| Visualization | Plotly | Interactive charts (hover, zoom, pan) |
| Authentication | streamlit-authenticator | Login/logout for dashboard access |
| Containerization | Docker | Packaging the entire app into a single container |
| Orchestration | Kubernetes | Deployment configuration for scaling |
| Version Control | Git + GitHub | Code tracking and collaboration |

---

## 8. Project File Structure

```
ml/
├── .streamlit/
│   └── config.toml                 # Dark theme configuration
├── src/
│   ├── data_preprocessing.py       # 🧹 Phase 2: 8-step data cleaning pipeline
│   ├── dashboard.py                # 📊 Phase 5: Streamlit dashboard (8 tabs + auth)
│   ├── database.py                 # SQLite schema setup
│   ├── data_generator.py           # Synthetic data generator (legacy, from Phase 1 of workshop)
│   ├── scheduler.py                # Automated periodic retraining scheduler
│   ├── api/
│   │   └── serve_models.py         # ⚡ Phase 4: FastAPI server (4 prediction endpoints)
│   ├── ml/
│   │   └── train_models.py         # 🧠 Phase 3: Multi-model training & comparison
│   ├── data/
│   │   ├── raw_superstore.csv      # 📥 Original real-world dataset (untouched)
│   │   ├── raw_sample.csv          # Sample of raw data for dashboard display
│   │   ├── cleaned_superstore.csv  # After all 8 preprocessing steps
│   │   ├── sales_data.csv          # Departmental table: Sales
│   │   ├── inventory_data.csv      # Departmental table: Inventory
│   │   ├── finance_data.csv        # Departmental table: Finance
│   │   ├── support_data.csv        # Departmental table: Customer Support
│   │   ├── logistics_data.csv      # Departmental table: Logistics
│   │   ├── ml_forecast.csv         # ML output: revenue predictions
│   │   ├── ml_anomalies.csv        # ML output: anomaly flags
│   │   ├── ml_clusters.csv         # ML output: cluster assignments
│   │   ├── preprocessing_report.json  # Before/after cleaning stats
│   │   ├── model_comparison.json   # All model metrics for comparison
│   │   └── retail.db               # SQLite database
│   ├── models/
│   │   ├── random_forest.pkl       # Best revenue model (saved)
│   │   ├── isolation_forest.pkl    # Anomaly detection model (saved)
│   │   ├── kmeans.pkl              # Best clustering model (saved)
│   │   ├── linear_regression.pkl   # Trendline model (saved)
│   │   └── cluster_mapping.pkl     # Dynamic cluster label mapping
│   └── crud/
│       └── operations.py           # Database CRUD utilities
├── main.py                         # CLI-based data viewer (legacy)
├── run.sh                          # 🚀 One-command startup script
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container build configuration
├── docker-compose.yml              # Multi-service orchestration
├── k8s-deployment.yaml             # Kubernetes deployment + service
└── README.md                       # This file
```

---

## 9. Login Credentials

The dashboard requires authentication. Use any of these:

| Username | Password | Role |
|----------|----------|------|
| `admin` | `admin123` | Admin User |
| `hod` | `hod2026` | HOD Faculty |
| `abc_retail` | `retail@2026` | ABC Retail Manager |

---

## 10. Key Talking Points for Presentation

If the guide asks questions, here are the key points to hit:

**"Where did you get your data?"**
> We used the Global Superstore Dataset — a real-world, publicly available retail dataset with 1000+ transactions across 147 countries. It includes actual sales, shipping, customer, and financial data.

**"What data cleaning did you do?"**
> We ran an 8-step preprocessing pipeline: duplicate removal, null imputation (806 missing postal codes filled), date parsing, outlier detection using IQR method (found 190 outlier rows across 5 columns), and feature engineering where we created 7 new derived columns like shipping delay days and profit margin.

**"Why did you choose these models?"**
> We didn't just choose them — we trained 15+ different models across 3 tasks and compared them on standard metrics. For revenue forecasting, we compared 6 models (Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting, SVR) and Random Forest won with R² = 0.93. For anomaly detection, we compared 3 models and chose Isolation Forest. For clustering, we compared 6 approaches and DBSCAN scored highest on Silhouette Score.

**"What about Docker/Kubernetes?"**
> The project is fully containerized. We have a Dockerfile, docker-compose.yml, and a Kubernetes deployment manifest. The Docker image was previously published to Docker Hub. We removed Docker locally due to storage constraints on the development machine, but the configuration files are all in place and the image was tested.

**"What does the dashboard show?"**
> 8 interactive tabs: Pipeline Overview (architecture), Data Preprocessing (before/after cleaning), Model Comparison (metrics + charts for all models), Anomaly Detection (scatter plot), Revenue Forecast (time series), Inventory Clusters (pie + scatter), Linear Regression Trendline, and a Live API + CRUD section for real-time predictions and database management.

**"Is there authentication?"**
> Yes, the dashboard has a GUI-based login system with multiple user accounts and session cookies.
