import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load artifacts (Churn Model + Cluster Model)
@st.cache_resource
def load_artifacts():
    # Churn artifacts
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    
    # Cluster artifacts (KNN wrapper for Agglomerative)
    cluster_knn = joblib.load('cluster_knn.pkl')
    cluster_scaler = joblib.load('cluster_scaler.pkl')
    cluster_features = joblib.load('cluster_features.pkl')
    
    return model, scaler, feature_names, cluster_knn, cluster_scaler, cluster_features

model, scaler, feature_names, cluster_knn, cluster_scaler, cluster_features = load_artifacts()

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("📞 Live Customer Churn Predictor")
st.markdown("Adjust the customer details below to see churn probability and customer segment.")

# Sidebar inputs
st.sidebar.header("Customer Profile")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 20.0, 120.0, 70.0)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
total_services = st.sidebar.slider("Number of additional services (0-6)", 0, 6, 2)

# --- CHURN PREDICTION (Existing Code) ---
contract_one = 1 if contract == "One year" else 0
contract_two = 1 if contract == "Two year" else 0
online_sec_yes = 1 if online_security == "Yes" else 0
tech_sup_yes = 1 if tech_support == "Yes" else 0

feature_dict = {}
for name in feature_names:
    if name == 'tenure': feature_dict[name] = tenure
    elif name == 'MonthlyCharges': feature_dict[name] = monthly_charges
    elif name == 'Contract_One year': feature_dict[name] = contract_one
    elif name == 'Contract_Two year': feature_dict[name] = contract_two
    elif name == 'OnlineSecurity_Yes': feature_dict[name] = online_sec_yes
    elif name == 'TechSupport_Yes': feature_dict[name] = tech_sup_yes
    elif name == 'TotalServices': feature_dict[name] = total_services
    else: feature_dict[name] = 0

if 'TotalCharges' in feature_names:
    feature_dict['TotalCharges'] = tenure * monthly_charges

X_input = pd.DataFrame([feature_dict])[feature_names]
X_scaled = scaler.transform(X_input)
prob = model.predict_proba(X_scaled)[0][1]

st.subheader("🔮 Churn Prediction")
st.metric("Risk Score", f"{prob:.0%}")

if prob > 0.5:
    st.error("⚠️ High churn risk - Offer loyalty discount.")
else:
    st.success("✅ Low churn risk - Consider upsell.")

# --- CLUSTER PREDICTION (NEW WEEK 10 INTEGRATION) ---
# Create input for clustering
cluster_input = {}
for name in cluster_features:
    if name == 'tenure': cluster_input[name] = tenure
    elif name == 'MonthlyCharges': cluster_input[name] = monthly_charges
    elif name == 'TotalCharges': cluster_input[name] = tenure * monthly_charges
    elif name == 'TotalServices': cluster_input[name] = total_services
    else: cluster_input[name] = 0

X_cluster_input = pd.DataFrame([cluster_input])[cluster_features]
X_cluster_scaled = cluster_scaler.transform(X_cluster_input)

# Use the KNN classifier to predict the cluster for this new point
cluster_label = cluster_knn.predict(X_cluster_scaled)[0]

# Map cluster to name
cluster_names = {
    0: "Loyal Low-Spenders",
    1: "High-Risk Newcomers",
    2: "Premium Loyal",
    3: "Mid-Tier Regulars"
}
segment_name = cluster_names.get(cluster_label, "Unknown")

st.markdown("---")
st.subheader("🏷️ Customer Segment")
st.info(f"**{segment_name}**")

st.caption("Model: Random Forest (Churn) + Agglomerative Clustering (Segment). Adjust sliders to see real-time predictions.")