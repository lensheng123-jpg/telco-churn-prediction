# Telco Customer Churn Prediction – A++ Project

## Overview
This project predicts customer churn for a telecom company using **Logistic Regression** and **Random Forest**. It includes exploratory data analysis, feature engineering, model training, evaluation, customer segmentation via **Agglomerative Clustering**, and an interactive **Streamlit dashboard**. The entire environment is containerised with **Docker**.

## Dataset
[Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- 7043 rows, 21 columns  
- No personally identifiable information

## Project Structure
- `churn_analysis.ipynb` – EDA, feature engineering, model training, clustering (Agglomerative), and saving the models.
- `app.py` – Interactive Streamlit dashboard that loads the trained models and predicts churn risk + customer segment in real time.
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
- **Logistic Regression** achieved **79.39% accuracy** and **0.8258 ROC‑AUC**, outperforming **Random Forest** (76.83% accuracy, 0.7878 ROC‑AUC).
- Key predictors: tenure, monthly charges, contract type, and number of services.
- **Customer Segmentation** identified 4 distinct segments: 
  - Mid-Tier Regulars (3,895 customers)
  - Premium Loyal (1,682 customers)
  - High-Risk Newcomers (824 customers)
  - Loyal Low-Spenders (631 customers)

## Recommendations
1. Convert month‑to‑month customers to annual contracts with a 10% discount.
2. Target high‑risk customers (tenure <12 months & monthly charges >$70) with retention offers.
3. Bundle online security and tech support to increase service adoption.

## Video Demo
Telco Customer Churn Prediction_Data analytics Part 1. https://youtu.be/pMF45e4Ay34

Telco Customer Churn Prediction_Data analytics Part 2. https://youtu.be/j3hrw7H3NMU

Comparison between Logistic Regression and Random Forest demo video. [https://youtu.be/nTS1H__3yho](https://youtu.be/Bo21vrFSXkc)





## Author
[Na Kuan Li] – [BIT_B2201F-2505002]
