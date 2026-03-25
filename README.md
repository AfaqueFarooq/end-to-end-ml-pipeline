# end-to-end-ml-pipeline with deployement

A full end-to-end machine learning web application that predicts a student's **mathematics score** based on demographic and academic factors.

**Live Demo → [ml-student-predictor.azurewebsites.net](https://student-performance-ml.azurewebsites.net](https://ml-student-predictor-deaqemb3efa2auge.francecentral-01.azurewebsites.net/))**

---

![App Preview](assets/preview.png)


---

## Features

- Predicts student math scores using 6 input features
- Trained and evaluated on 9+ regression models
- Best model automatically selected based on R² score
- Clean modular ML pipeline (ingestion → transformation → training)
- Flask web app with a modern dark UI
- Deployed on **Azure App Service** with CI/CD via GitHub Actions

---

## ML Pipeline

```
Raw Data (CSV)
    ↓
Data Ingestion        → splits into train/test, saves to artifacts/
    ↓
Data Transformation   → handles missing values, encodes categories, scales features
    ↓
Model Training        → trains 9 models, evaluates, saves best model
    ↓
Predict Pipeline      → loads saved model, returns prediction
    ↓
Flask Web App         → serves prediction via UI
```

---

## Models Evaluated

| Model | Description |
|---|---|
| Linear Regression | Baseline model |
| Lasso | L1 regularization |
| Ridge | L2 regularization |
| K-Neighbors Regressor | Distance-based |
| Decision Tree | Tree-based |
| Random Forest | Ensemble bagging |
| XGBoost | Gradient boosting |
| CatBoost | Categorical boosting |
| AdaBoost | Adaptive boosting |

> Best model is selected automatically based on R² score on the test set.

---

## Tech Stack

**ML & Data**
- Python, Pandas, NumPy
- Scikit-learn, XGBoost, CatBoost
- Dill (model serialization)

**Web**
- Flask
- HTML/CSS (custom dark UI)

**DevOps**
- GitHub Actions (CI/CD)
- Azure App Service (deployment)

---

## 🏃 Run Locally

### Prerequisites

```bash
# Mac only — required for XGBoost
brew install libomp
```

### Setup

```bash
# Clone the repo
git clone https://github.com/AfaqueFarooq/end-to-end-ml-pipeline.git
cd end-to-end-ml-pipeline

# Create and activate virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```


---

## Deployment

This project is deployed on **Azure App Service** with automatic deployments triggered on every push to `main` via GitHub Actions.

---

