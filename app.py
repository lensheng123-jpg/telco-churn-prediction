import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Churn Predictor Pro", layout="wide")

# --- DARK MODE TOGGLE ---
st.sidebar.markdown("---")
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

# --- CSS (Dark Mode) ---
if dark_mode:
    st.markdown(
        """
        <style>
        .stApp, .stSidebar, .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, 
        .stMarkdown h3, .stMarkdown h4, .stSidebar .stSidebarContent, 
        .stSidebar .stSidebarContent .stMarkdown, .stSidebar .stSidebarContent label,
        .stMetric, .stMetric label, .stMetric .stMetricValue, .stMetric .stMetricLabel,
        .stDataFrame, .stDataFrame table, .stDataFrame thead th, .stDataFrame tbody td,
        .stExpander, .stExpander .stExpanderHeader, .stAlert, .stAlert .stAlertContent,
        .stButton button, .stDownloadButton button, .stCaption, .stColumn .stMarkdown,
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
        .stCheckbox label, .stToggle label, .stSelectbox label, .stSlider label, .stNumberInput label,
        .stSidebar .stHeader, .stSidebar .stSubheader,
        .st-emotion-cache-1r6slb0, .st-emotion-cache-16txtl3 {
            color: #FFFFFF !important;
        }
        .stSidebar { background-color: #1E1E1E !important; }
        .stApp { background-color: #0E1117 !important; }
        .stDataFrame thead th { background-color: #2E2E2E !important; color: #FFFFFF !important; }
        .stDataFrame tbody tr { background-color: #1E1E1E !important; }
        .stDataFrame tbody tr:nth-child(even) { background-color: #2A2A2A !important; }
        .stDataFrame tbody td { color: #FFFFFF !important; background-color: transparent !important; }
        .stButton button, .stDownloadButton button { background-color: #2E2E2E !important; border: 1px solid #555555 !important; color: #FFFFFF !important; }
        .stButton button:hover, .stDownloadButton button:hover { background-color: #3E3E3E !important; color: #FFFFFF !important; }
        .stMetric .stMetricDelta { color: #AAAAAA !important; }
        .stCaption { color: #AAAAAA !important; }
        .stAlert { background-color: #2E2E2E !important; }
        .stAlert .stAlertContent { color: #FFFFFF !important; }
        .stToggle label { color: #FFFFFF !important; }
        .stMetric .stMetricLabel { color: #FFFFFF !important; }
        .stMetric .stMetricValue { color: #FFFFFF !important; }
        .stExpander { background-color: #1E1E1E !important; border: 1px solid #333333 !important; border-radius: 8px !important; }
        .stExpander .stExpanderHeader { color: #FFFFFF !important; background-color: #2A2A2A !important; border-radius: 8px 8px 0 0 !important; }
        .stExpander .stExpanderContent { background-color: #1E1E1E !important; color: #FFFFFF !important; border-radius: 0 0 8px 8px !important; }
        .stExpander .stExpanderContent .stMarkdown { color: #FFFFFF !important; }
        .stSubheader { color: #FFFFFF !important; }

        /* ===== FIX: Model Selector (Radio) Text in Dark Mode ===== */
        [data-testid="stRadio"] label {
            color: #FFFFFF !important;
        }
        [data-testid="stRadio"] div[role="radiogroup"] p {
            color: #FFFFFF !important;
        }
        [data-testid="stRadio"] div[role="radiogroup"] div {
            color: #FFFFFF !important;
        }
        [data-testid="stRadio"] input[type="radio"] {
            accent-color: #4FC3F7;
        }
        .stSidebar .stCaption, .stCaption {
            color: #CCCCCC !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .stApp { background-color: #F8F9FA; }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    rf_model = joblib.load('churn_model.pkl')   # Random Forest (existing)
    lr_model = joblib.load('lr_model.pkl')      # Logistic Regression (newly saved)
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    cluster_knn = joblib.load('cluster_knn.pkl')
    cluster_scaler = joblib.load('cluster_scaler.pkl')
    cluster_features = joblib.load('cluster_features.pkl')
    return rf_model, lr_model, scaler, feature_names, cluster_knn, cluster_scaler, cluster_features

rf_model, lr_model, scaler, feature_names, cluster_knn, cluster_scaler, cluster_features = load_artifacts()

# --- STATIC DATA ---
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

# --- UI TITLE ---
st.title("📞 Telco Churn Predictor Pro")
st.markdown("**Enterprise-grade customer churn prediction and segmentation**")

# --- SIDEBAR ---
st.sidebar.header("👤 Customer Profile")

# Initialize session state
if 'tenure' not in st.session_state:
    st.session_state.tenure = 12
    st.session_state.monthly_charges = 70.0
    st.session_state.contract = "Month-to-month"
    st.session_state.online_security = "No"
    st.session_state.tech_support = "No"
    st.session_state.total_services = 2
    st.session_state.reset = False

if st.session_state.reset:
    st.session_state.tenure = 12
    st.session_state.monthly_charges = 70.0
    st.session_state.contract = "Month-to-month"
    st.session_state.online_security = "No"
    st.session_state.tech_support = "No"
    st.session_state.total_services = 2
    st.session_state.reset = False

tenure = st.sidebar.slider("Tenure (months)", 0, 72, key='tenure')
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 20.0, 120.0, key='monthly_charges')
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key='contract')
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"], key='online_security')
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"], key='tech_support')
total_services = st.sidebar.slider("Additional Services", 0, 6, key='total_services')

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset to Default"):
    st.session_state.reset = True
    st.rerun()

# --- MODEL SELECTOR (NEW!) ---
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Model Selection")
model_choice = st.sidebar.radio(
    "Choose Prediction Model:",
    ["Logistic Regression ★ Best", "Random Forest (Prototype)"],
    index=0  # Default to Logistic Regression
)
st.sidebar.caption("Logistic Regression achieved 0.8258 ROC-AUC, outperforming Random Forest (0.7878)")

# --- DATA INTEGRITY ---
if tenure < 3 and monthly_charges > 70:
    st.sidebar.error("🔴 HIGH RISK: New customer, high spend!")

# --- PREDICTIONS (Dynamic based on Model Selection) ---
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

# --- SELECT MODEL ---
if model_choice == "Logistic Regression ★ Best":
    model = lr_model
    model_name = "Logistic Regression"
    model_badge = "⭐"
else:
    model = rf_model
    model_name = "Random Forest"
    model_badge = "🔬"

prob = model.predict_proba(X_scaled)[0][1]

# --- PROFESSIONAL RISK GAUGE ---
st.subheader(f"🔮 Churn Prediction ({model_badge} {model_name})")
col_g1, col_g2 = st.columns([3, 1])

with col_g1:
    risk_percent = prob * 100
    st.markdown(f"### Risk Score: **{risk_percent:.0f}%**")
    
    if risk_percent < 30:
        bar_color = "green"
        status = "✅ Low Risk"
    elif risk_percent < 60:
        bar_color = "orange"
        status = "⚠️ Medium Risk"
    else:
        bar_color = "red"
        status = "🔴 High Risk"
    
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
    confidence_score = abs(prob - 0.5) * 2
    if confidence_score > 0.7:
        confidence_label = "🔥 High"
    elif confidence_score > 0.4:
        confidence_label = "📊 Moderate"
    else:
        confidence_label = "❓ Low"
    st.metric("Confidence", confidence_label, delta=None)

# --- RISK DRIVER BREAKDOWN (Dynamic for LR or RF) ---
st.markdown("---")
st.subheader(f"📊 Risk Driver Breakdown ({model_name})")
driver_data = []

if model_choice == "Logistic Regression ★ Best":
    # Use Logistic Regression Coefficients
    coef = model.coef_[0]
    for i, name in enumerate(feature_names):
        if name in ['tenure', 'MonthlyCharges', 'TotalServices', 'TotalCharges']:
            impact = coef[i] * 10  # Scale for readability
            direction = "⬆️ Increases Risk" if impact > 0.1 else "⬇️ Decreases Risk" if impact < -0.1 else "➖ Neutral"
            driver_data.append({
                'Feature': name,
                'Coefficient': round(impact, 3),
                'Direction': direction
            })
else:
    # Use Random Forest Feature Importance
    importance_rf = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    })
    for _, row in importance_rf.iterrows():
        if row['feature'] in ['tenure', 'MonthlyCharges', 'TotalServices', 'TotalCharges']:
            impact = row['importance']
            direction = "⬆️ Increases Risk" if impact > 0.05 else "⬇️ Decreases Risk" if impact < 0.02 else "➖ Neutral"
            driver_data.append({
                'Feature': row['feature'],
                'Importance': round(impact, 3),
                'Direction': direction
            })

st.dataframe(pd.DataFrame(driver_data), use_container_width=True)

# --- RETENTION SIMULATOR (Dynamic) ---
st.markdown("---")
st.subheader("💡 Retention Simulator")
st.caption("See how risk changes if we improve the customer's profile.")

col_s1, col_s2, col_s3 = st.columns(3)

# Simulation 1: Tenure +24 months
tenure_whatif = min(tenure + 24, 72)
feature_dict_tenure = feature_dict.copy()
feature_dict_tenure['tenure'] = tenure_whatif
if 'TotalCharges' in feature_names:
    feature_dict_tenure['TotalCharges'] = tenure_whatif * monthly_charges
X_tenure = pd.DataFrame([feature_dict_tenure])[feature_names]
X_tenure_scaled = scaler.transform(X_tenure)
prob_tenure = model.predict_proba(X_tenure_scaled)[0][1]
tenure_improvement = (prob - prob_tenure) * 100

with col_s1:
    delta_color = "normal" if tenure_improvement > 0 else "inverse"
    st.metric("If Tenure +24 months", f"{prob_tenure:.0%}", delta=f"↓ -{tenure_improvement:.0f}%", delta_color=delta_color)

# Simulation 2: +3 Services
services_whatif = min(total_services + 3, 6)
feature_dict_services = feature_dict.copy()
feature_dict_services['TotalServices'] = services_whatif
X_services = pd.DataFrame([feature_dict_services])[feature_names]
X_services_scaled = scaler.transform(X_services)
prob_services = model.predict_proba(X_services_scaled)[0][1]
services_improvement = (prob - prob_services) * 100

with col_s2:
    delta_color = "normal" if services_improvement > 0 else "inverse"
    st.metric("If +3 Services", f"{prob_services:.0%}", delta=f"↓ -{services_improvement:.0f}%", delta_color=delta_color)

# Simulation 3: 2-Year Contract
feature_dict_contract = feature_dict.copy()
feature_dict_contract['Contract_Two year'] = 1
feature_dict_contract['Contract_One year'] = 0
X_contract = pd.DataFrame([feature_dict_contract])[feature_names]
X_contract_scaled = scaler.transform(X_contract)
prob_contract = model.predict_proba(X_contract_scaled)[0][1]
contract_improvement = (prob - prob_contract) * 100

with col_s3:
    delta_color = "normal" if contract_improvement > 0 else "inverse"
    st.metric("If 2-Year Contract", f"{prob_contract:.0%}", delta=f"↓ -{contract_improvement:.0f}%", delta_color=delta_color)

st.caption("💡 Simulate different customer profiles by adjusting the sliders above.")

# --- SEGMENT PREDICTION (Unchanged) ---
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
    st.markdown(f'<div style="background-color: #1E3A5F; padding: 12px; border-radius: 8px; border-left: 5px solid #4FC3F7; color: #B3E5FC; font-weight: bold; font-size: 18px;">🏷️ {segment_name}</div>', unsafe_allow_html=True)

# --- BENCHMARKING (Unchanged) ---
st.markdown("---")
st.subheader("📈 Customer Benchmarking")
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

# --- DOWNLOAD REPORT (Unchanged) ---
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
    st.download_button(label="📥 Download Profile (CSV)", data=csv_data,
                       file_name=f"customer_profile_{tenure}mo_{segment_name.replace(' ', '_')}.csv",
                       mime="text/csv", key="download_csv")

# --- DETAILED INSIGHTS (Dynamic for LR or RF) ---
with st.expander("📊 Model Insights & Analytics"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader(f"Feature Impact ({model_name})")
        fig, ax = plt.subplots(figsize=(8, 4))
        if dark_mode:
            fig.patch.set_facecolor('#1E1E1E')
            ax.set_facecolor('#1E1E1E')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
        
        if model_choice == "Logistic Regression ★ Best":
            # Show Logistic Regression Coefficients
            coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': model.coef_[0]})
            coef_df = coef_df.sort_values('Coefficient', ascending=False).head(10)
            sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='coolwarm', ax=ax)
            ax.set_title('Top 10 Feature Coefficients (LR)')
            ax.set_xlabel('Coefficient (Higher = Increases Churn)')
        else:
            # Show Random Forest Feature Importance
            imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': rf_model.feature_importances_})
            imp_df = imp_df.sort_values('Importance', ascending=False).head(10)
            sns.barplot(x='Importance', y='Feature', data=imp_df, palette='Blues_d', ax=ax)
            ax.set_title('Top 10 Feature Importances (RF)')
            ax.set_xlabel('Importance Score')
        st.pyplot(fig)
        plt.close(fig)
    
    with col_b:
        st.subheader("Customer Segment Distribution")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        if dark_mode:
            fig2.patch.set_facecolor('#1E1E1E')
            ax2.set_facecolor('#1E1E1E')
            ax2.tick_params(colors='white')
            ax2.title.set_color('white')
        colors = ['#2E86C1', '#28B463', '#F39C12', '#5B2C6F']
        ax2.pie(cluster_profiles['Count'], labels=cluster_profiles['Segment'], 
                autopct='%1.1f%%', startangle=90, colors=colors, textprops={'color': 'white' if dark_mode else 'black'})
        ax2.axis('equal')
        ax2.set_title('Customer Segment Distribution', color='white' if dark_mode else 'black')
        st.pyplot(fig2)
        plt.close(fig2)
    
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Segment Profiles")
        st.dataframe(cluster_profiles.style.background_gradient(cmap='Blues'), use_container_width=True)
    with col_d:
        st.subheader("Model Performance Comparison")
        metrics_df = pd.DataFrame({
            'Model': ['Logistic Regression ★ Best', 'Random Forest'],
            'Accuracy': [0.7939, 0.7683],
            'ROC-AUC': [0.8258, 0.7878]
        })
        st.dataframe(metrics_df.style.highlight_max(subset=['Accuracy', 'ROC-AUC'], color='lightgreen'), use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.caption(f"Built with ❤️ using Streamlit | Current Model: {model_name} | Containerised with Docker")
