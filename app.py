import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Churn Predictor Pro", layout="wide")

# --- DARK MODE TOGGLE (Must be at the top) ---
st.sidebar.markdown("---")
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

if dark_mode:
    st.markdown(
        """
        <style>
        /* === FORCE EVERYTHING WHITE IN DARK MODE === */
        .stApp, .stSidebar, .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, 
        .stMarkdown h3, .stMarkdown h4, .stSidebar .stSidebarContent, 
        .stSidebar .stSidebarContent .stMarkdown, .stSidebar .stSidebarContent label,
        .stSidebar .stSidebarContent .stMarkdown h1,
        .stSidebar .stSidebarContent .stMarkdown h2,
        .stSidebar .stSidebarContent .stMarkdown h3,
        .stSidebar .stSidebarContent .stMarkdown h4,
        .stMetric, .stMetric label, .stMetric .stMetricValue, .stMetric .stMetricLabel,
        .stDataFrame, .stDataFrame table, .stDataFrame thead th, .stDataFrame tbody td,
        .stExpander, .stExpander .stExpanderHeader, .stAlert, .stAlert .stAlertContent,
        .stButton button, .stDownloadButton button, .stCaption, .stColumn .stMarkdown,
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
        .stCheckbox label, .stToggle label, .stSelectbox label, .stSlider label, .stNumberInput label,
        .stSidebar .stHeader, .stSidebar .stSubheader,
        .st-emotion-cache-1r6slb0, .st-emotion-cache-16txtl3,
        .stSidebar .stSidebarContent .stHeader,
        .stSidebar .stSidebarContent .stSubheader {
            color: #FFFFFF !important;
        }
        
        /* Sidebar background */
        .stSidebar {
            background-color: #1E1E1E !important;
        }
        .stApp {
            background-color: #0E1117 !important;
        }
        
        /* Dataframe headers and cells - DARK ROWS */
        .stDataFrame thead th {
            background-color: #2E2E2E !important;
            color: #FFFFFF !important;
        }
        .stDataFrame tbody tr {
            background-color: #1E1E1E !important;
        }
        .stDataFrame tbody tr:nth-child(even) {
            background-color: #2A2A2A !important;
        }
        .stDataFrame tbody tr:nth-child(odd) {
            background-color: #1E1E1E !important;
        }
        .stDataFrame tbody td {
            color: #FFFFFF !important;
            background-color: transparent !important;
        }
        
        /* Buttons */
        .stButton button, .stDownloadButton button {
            background-color: #2E2E2E !important;
            border: 1px solid #555555 !important;
            color: #FFFFFF !important;
        }
        .stButton button:hover, .stDownloadButton button:hover {
            background-color: #3E3E3E !important;
            color: #FFFFFF !important;
        }
        
        /* Metric deltas */
        .stMetric .stMetricDelta {
            color: #AAAAAA !important;
        }
        
        /* Captions */
        .stCaption {
            color: #AAAAAA !important;
        }
        
        /* Info/Success/Error boxes */
        .stAlert {
            background-color: #2E2E2E !important;
        }
        .stAlert .stAlertContent {
            color: #FFFFFF !important;
        }

        /* === DARK MODE TOGGLE LABEL - FIX === */
        .stToggle label {
            color: #FFFFFF !important;
        }
        div[data-testid="stCheckbox"] label {
            color: #FFFFFF !important;
        }
        div[data-testid="stToggle"] label {
            color: #FFFFFF !important;
        }
        .stSidebar .stToggle label {
            color: #FFFFFF !important;
        }
        .stSidebar .stSidebarContent .stToggle label {
            color: #FFFFFF !important;
        }
        .stSidebar .stSidebarContent label {
            color: #FFFFFF !important;
        }
        .st-emotion-cache-1r6slb0 .stToggle label {
            color: #FFFFFF !important;
        }
        .st-emotion-cache-16txtl3 label {
            color: #FFFFFF !important;
        }
        
        /* === EXPANDER STYLING - FIXED FOR DARK MODE === */
        .stExpander {
            background-color: #1E1E1E !important;
            color: #FFFFFF !important;
            border: 1px solid #333333 !important;
            border-radius: 8px !important;
        }
        .stExpander .stExpanderHeader {
            color: #FFFFFF !important;
            background-color: #2A2A2A !important;
            border-radius: 8px 8px 0 0 !important;
            padding: 12px !important;
        }
        .stExpander .stExpanderHeader:hover {
            background-color: #333333 !important;
        }
        /* EXPANDER CONTENT AREA - FORCE DARK */
        .stExpander .stExpanderContent {
            background-color: #1E1E1E !important;
            color: #FFFFFF !important;
            padding: 10px !important;
            border-radius: 0 0 8px 8px !important;
        }
        .stExpander .stExpanderContent .stMarkdown {
            color: #FFFFFF !important;
        }
        .stExpander .stExpanderContent .stColumn .stMarkdown {
            color: #FFFFFF !important;
        }
        .stExpander .stExpanderContent .stSubheader {
            color: #FFFFFF !important;
        }
        
        /* === SUBHEADER INSIDE EXPANDER === */
        .stSubheader {
            color: #FFFFFF !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #F8F9FA;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    cluster_knn = joblib.load('cluster_knn.pkl')
    cluster_scaler = joblib.load('cluster_scaler.pkl')
    cluster_features = joblib.load('cluster_features.pkl')
    return model, scaler, feature_names, cluster_knn, cluster_scaler, cluster_features

model, scaler, feature_names, cluster_knn, cluster_scaler, cluster_features = load_artifacts()

# --- STATIC DATA FOR VISUALISATIONS ---
feature_importance = pd.DataFrame({
    'Feature': ['MonthlyCharges', 'TotalCharges', 'tenure', 'TotalServices', 
                'TenureGroup', 'PaperlessBilling', 'SeniorCitizen', 'Partner', 'Dependents'],
    'Importance': [0.32, 0.26, 0.18, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
})
cluster_profiles = pd.DataFrame({
    'Segment': ['Mid-Tier Regulars', 'Premium Loyal', 'High-Risk Newcomers', 'Loyal Low-Spenders'],
    'Avg Tenure': [20.13, 33.30, 55.84, 15.0],
    'Avg Charges': [49.53, 79.82, 85.91, 35.0],
    'Count': [3895, 1682, 824, 631]
})
metrics_df = pd.DataFrame({
    'Model': ['Logistic Regression ★ Best', 'Random Forest'],
    'Accuracy': [0.7939, 0.7683],
    'ROC-AUC': [0.8258, 0.7878]
})

# --- UI TITLE ---
st.title("📞 Telco Churn Predictor Pro")
st.markdown("**Enterprise-grade customer churn prediction and segmentation**")

# --- SIDEBAR ---
st.sidebar.header("👤 Customer Profile")

# Initialize session state defaults
if 'tenure' not in st.session_state:
    st.session_state.tenure = 12
    st.session_state.monthly_charges = 70.0
    st.session_state.contract = "Month-to-month"
    st.session_state.online_security = "No"
    st.session_state.tech_support = "No"
    st.session_state.total_services = 2
    st.session_state.reset = False

# --- RESET HANDLER (RUNS BEFORE WIDGETS) ---
if st.session_state.reset:
    st.session_state.tenure = 12
    st.session_state.monthly_charges = 70.0
    st.session_state.contract = "Month-to-month"
    st.session_state.online_security = "No"
    st.session_state.tech_support = "No"
    st.session_state.total_services = 2
    st.session_state.reset = False

# --- WIDGETS ---
tenure = st.sidebar.slider("Tenure (months)", 0, 72, key='tenure')
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 20.0, 120.0, key='monthly_charges')
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key='contract')
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"], key='online_security')
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"], key='tech_support')
total_services = st.sidebar.slider("Additional Services", 0, 6, key='total_services')

# --- RESET BUTTON ---
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset to Default"):
    st.session_state.reset = True
    st.rerun()

# --- DATA INTEGRITY CHECK ---
if tenure < 3 and monthly_charges > 70:
    st.sidebar.error("🔴 HIGH RISK: New customer, high spend!")

# --- PREDICTIONS ---
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

# --- PROFESSIONAL RISK GAUGE ---
st.subheader("🔮 Churn Prediction")

# Create columns for the gauge and the label
col_g1, col_g2 = st.columns([3, 1])

with col_g1:
    risk_percent = prob * 100
    st.markdown(f"### Risk Score: **{risk_percent:.0f}%**")
    
    # Color-coded progress bar
    if risk_percent < 30:
        bar_color = "green"
        status = "✅ Low Risk"
    elif risk_percent < 60:
        bar_color = "orange"
        status = "⚠️ Medium Risk"
    else:
        bar_color = "red"
        status = "🔴 High Risk"
    
    # HTML/CSS for a thick, modern progress bar
    st.markdown(
        f"""
        <div style="background-color: #e0e0e0; border-radius: 10px; height: 25px; width: 100%;">
            <div style="background-color: {bar_color}; border-radius: 10px; height: 25px; width: {risk_percent:.1f}%; 
                        display: flex; align-items: center; justify-content: flex-end; padding-right: 10px;
                        color: white; font-weight: bold; font-size: 14px; transition: width 1s ease-in-out;">
                {risk_percent:.0f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption(f"**{status}** - {'Consider upsell' if risk_percent < 50 else 'Immediate retention action required'}")

with col_g2:
    # Calculate confidence based on distance from 0.5 (uncertainty boundary)
    confidence_score = abs(prob - 0.5) * 2  # 0 = uncertain, 1 = highly certain
    if confidence_score > 0.7:
        confidence_label = "🔥 High"
    elif confidence_score > 0.4:
        confidence_label = "📊 Moderate"
    else:
        confidence_label = "❓ Low"
    
    st.metric("Confidence", confidence_label, delta=None)

# --- RISK DRIVER BREAKDOWN ---
st.markdown("---")
st.subheader("📊 Risk Driver Breakdown")
driver_data = []
for name in ['tenure', 'MonthlyCharges', 'TotalServices']:
    feature_dict_alt = feature_dict.copy()
    if name == 'tenure':
        feature_dict_alt[name] = max(0, tenure - 12)
    elif name == 'MonthlyCharges':
        feature_dict_alt[name] = max(20, monthly_charges - 20)
    elif name == 'TotalServices':
        feature_dict_alt[name] = max(0, total_services - 2)
    
    X_alt = pd.DataFrame([feature_dict_alt])[feature_names]
    X_alt_scaled = scaler.transform(X_alt)
    prob_alt = model.predict_proba(X_alt_scaled)[0][1]
    impact = (prob - prob_alt) * 100
    
    driver_data.append({
        'Feature': name.title(),
        'Impact on Risk %': round(impact, 2),
        'Direction': '⬆️ Increases Risk' if impact > 3 else '⬇️ Decreases Risk' if impact < -3 else '➖ Neutral'
    })
st.dataframe(pd.DataFrame(driver_data), use_container_width=True)

# --- RETENTION SIMULATOR WITH VISUAL TREND ---
st.markdown("---")
st.subheader("💡 Retention Simulator")
col_s1, col_s2 = st.columns(2)

# Simulate: Convert to 2-year contract
feature_dict_whatif = feature_dict.copy()
feature_dict_whatif['Contract_Two year'] = 1
feature_dict_whatif['Contract_One year'] = 0
X_whatif = pd.DataFrame([feature_dict_whatif])[feature_names]
X_whatif_scaled = scaler.transform(X_whatif)
prob_whatif = model.predict_proba(X_whatif_scaled)[0][1]
improvement = (prob - prob_whatif) * 100

with col_s1:
    st.metric("Current Risk", f"{prob:.0%}")

with col_s2:
    # Determine the trend category
    if improvement > 5:
        trend_label = "🔥 Best (Drops Significantly)"
        delta_color = "normal"
    elif improvement > 1:
        trend_label = "✅ Good (Drops Slightly)"
        delta_color = "normal"
    elif improvement > -1:
        trend_label = "➖ Neutral (No Change)"
        delta_color = "off"
    elif improvement > -5:
        trend_label = "⚠️ Poor (Drops Very Little)"
        delta_color = "inverse"
    else:
        trend_label = "🚨 Worst (Risk Increases)"
        delta_color = "inverse"
    
    st.metric(
        "If 2-Year Contract",
        f"{prob_whatif:.0%}",
        delta=f"{trend_label} ({improvement:.0f}%)",
        delta_color=delta_color
    )

st.caption("💡 Simulate different customer profiles by adjusting the sliders above.")

# --- SEGMENT PREDICTION ---
cluster_input = {}
for name in cluster_features:
    if name == 'tenure': cluster_input[name] = tenure
    elif name == 'MonthlyCharges': cluster_input[name] = monthly_charges
    elif name == 'TotalCharges': cluster_input[name] = tenure * monthly_charges
    elif name == 'TotalServices': cluster_input[name] = total_services
    else: cluster_input[name] = 0

X_cluster_input = pd.DataFrame([cluster_input])[cluster_features]
X_cluster_scaled = cluster_scaler.transform(X_cluster_input)
cluster_label = cluster_knn.predict(X_cluster_scaled)[0]

cluster_names = {0: "Loyal Low-Spenders", 1: "High-Risk Newcomers", 2: "Premium Loyal", 3: "Mid-Tier Regulars"}
segment_name = cluster_names.get(cluster_label, "Unknown")

st.markdown("---")
col_seg, col_seg2 = st.columns(2)
with col_seg:
    st.subheader("🏷️ Customer Segment")
    # Custom light blue styled segment label
    st.markdown(f'<div style="background-color: #1E3A5F; padding: 12px; border-radius: 8px; border-left: 5px solid #4FC3F7; color: #B3E5FC; font-weight: bold; font-size: 18px;">🏷️ {segment_name}</div>', unsafe_allow_html=True)
# --- BENCHMARKING (Compare to Average) ---
st.markdown("---")
st.subheader("📈 Customer Benchmarking")

# Define "Average" customer profile (based on your dataset)
avg_tenure = 32.4
avg_charges = 64.8
avg_services = 2.5

col_b1, col_b2, col_b3 = st.columns(3)

with col_b1:
    tenure_diff = tenure - avg_tenure
    if tenure_diff > 0:
        st.success(f"✅ +{tenure_diff:.0f} months **above** average tenure")
    else:
        st.error(f"⚠️ {abs(tenure_diff):.0f} months **below** average tenure")

with col_b2:
    charges_diff = monthly_charges - avg_charges
    if charges_diff > 0:
        st.error(f"⚠️ ${charges_diff:.0f} **higher** than average charges")
    else:
        st.success(f"✅ ${abs(charges_diff):.0f} **lower** than average charges")

with col_b3:
    services_diff = total_services - avg_services
    if services_diff > 0:
        st.success(f"✅ +{services_diff:.0f} more services than average")
    else:
        st.error(f"⚠️ {abs(services_diff):.0f} fewer services than average")

# --- DOWNLOAD REPORT ---
with col_seg2:
    st.subheader("📄 Export Report")
    
    def generate_report():
        report_data = {
            'Metric': ['Tenure', 'Monthly Charges', 'Contract', 'Online Security', 
                       'Tech Support', 'Services', 'Risk Score', 'Segment'],
            'Value': [tenure, f"${monthly_charges:.2f}", contract, online_security, 
                      tech_support, total_services, f"{prob:.0%}", segment_name]
        }
        report_df = pd.DataFrame(report_data)
        csv_buffer = io.StringIO()
        report_df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()
    
    csv_data = generate_report()
    st.download_button(
        label="📥 Download Profile (CSV)",
        data=csv_data,
        file_name=f"customer_profile_{tenure}mo_{segment_name.replace(' ', '_')}.csv",
        mime="text/csv",
        key="download_csv"
    )

# --- DETAILED INSIGHTS (Expandable) ---
with st.expander("📊 Model Insights & Analytics"):
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='Blues_d', ax=ax)
        ax.set_title('Top Predictors of Churn')
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('')
        st.pyplot(fig)
        plt.close(fig)
    
    with col_b:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        colors = ['#2E86C1', '#28B463', '#F39C12', '#5B2C6F']
        ax2.pie(cluster_profiles['Count'], labels=cluster_profiles['Segment'], 
                autopct='%1.1f%%', startangle=90, colors=colors)
        ax2.axis('equal')
        ax2.set_title('Customer Segment Distribution')
        st.pyplot(fig2)
        plt.close(fig2)
    
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Segment Profiles")
        st.dataframe(cluster_profiles.style.background_gradient(cmap='Blues'), use_container_width=True)
    
    with col_d:
        st.subheader("Model Performance Comparison")
        st.dataframe(metrics_df.style.highlight_max(subset=['Accuracy', 'ROC-AUC'], color='lightgreen'), use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Random Forest (Churn) + Agglomerative Clustering (Segment) | Containerised with Docker")
