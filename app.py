import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("📞 Live Customer Churn Predictor")
st.markdown("Adjust the customer details below to see churn probability in real time.")

# Sidebar inputs (simplified for demo)
st.sidebar.header("Customer Profile")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 20.0, 120.0, 70.0)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
total_services = st.sidebar.slider("Number of additional services (0-6)", 0, 6, 2)

# Create a dictionary with all feature names (fill missing with 0)
contract_one = 1 if contract == "One year" else 0
contract_two = 1 if contract == "Two year" else 0
online_sec_yes = 1 if online_security == "Yes" else 0
tech_sup_yes = 1 if tech_support == "Yes" else 0

# Build feature dictionary dynamically
feature_dict = {}
for name in feature_names:
    if name == 'tenure':
        feature_dict[name] = tenure
    elif name == 'MonthlyCharges':
        feature_dict[name] = monthly_charges
    elif name == 'Contract_One year':
        feature_dict[name] = contract_one
    elif name == 'Contract_Two year':
        feature_dict[name] = contract_two
    elif name == 'OnlineSecurity_Yes':
        feature_dict[name] = online_sec_yes
    elif name == 'TechSupport_Yes':
        feature_dict[name] = tech_sup_yes
    elif name == 'TotalServices':
        feature_dict[name] = total_services
    else:
        # For all other features (e.g., PaymentMethod_Electronic check, etc.), set to 0
        feature_dict[name] = 0

# Add derived features that might be needed
if 'TotalCharges' in feature_names:
    feature_dict['TotalCharges'] = tenure * monthly_charges

# Create DataFrame and scale
X_input = pd.DataFrame([feature_dict])[feature_names]  # ensure correct order
X_scaled = scaler.transform(X_input)
prob = model.predict_proba(X_scaled)[0][1]

st.subheader("🔮 Churn Probability")
st.metric("Risk Score", f"{prob:.0%}")

if prob > 0.5:
    st.error("⚠️ High churn risk")
    st.write("**Recommendation:** Offer loyalty discount or free tech support.")
else:
    st.success("✅ Low churn risk")
    st.write("**Recommendation:** Consider upsell or referral program.")

st.markdown("---")
st.caption("Model: Random Forest trained on Telco Customer Churn dataset. Adjust sliders to see real-time predictions.")