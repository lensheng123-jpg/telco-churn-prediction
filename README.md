# Telco Customer Churn Prediction – A++ Project

## Overview
This project predicts customer churn for a telecom company using Logistic Regression and Random Forest. It includes exploratory data analysis, feature engineering, model training, evaluation, and an interactive Streamlit dashboard. The entire environment is containerised with Docker.

## Dataset
[Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- 7043 rows, 21 columns  
- No personally identifiable information

## Project Structure
- `churn_analysis.ipynb` – EDA, feature engineering, model training, and saving the model.
- `app.py` – Interactive Streamlit dashboard that loads the trained model and predicts in real time.
- `Dockerfile`, `docker-compose.yml`, `requirements.txt` – Containerisation.
- `figures/` – All visualisations from the notebook.
- `data/` – The raw CSV file.
- `report.pdf` – Full project report (max 10 pages).

## How to Run

### Locally (without Docker)
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook: `jupyter notebook churn_analysis.ipynb` (run all cells).
4. Launch the dashboard: `streamlit run app.py`
5. Open `http://localhost:8501` in your browser.

### With Docker
1. Clone the repository.
2. Build and run: `docker-compose up --build`
3. Open `http://localhost:8501`

## Results
- **Random Forest** achieved **81% accuracy** and **0.85 ROC‑AUC**, outperforming Logistic Regression.
- Key predictors: tenure, monthly charges, contract type, and number of services.

## Recommendations
1. Convert month‑to‑month customers to annual contracts with a 10% discount.
2. Target high‑risk customers (tenure <12 months & monthly charges >$70) with retention offers.
3. Bundle online security and tech support to increase service adoption.

## Video Demo
[Watch the presentation](https://youtu.be/your-link-here)

## Author
[Your Name] – [Your Student ID]
