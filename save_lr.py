import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load your cleaned data
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Do the same cleaning as your notebook
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df[col] = df[col].map({'Yes':1, 'No':0})

cat_cols = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Feature engineering
service_cols = ['OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes',
                'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes']
df['TotalServices'] = df[service_cols].sum(axis=1)
df['TenureGroup'] = pd.cut(df['tenure'], bins=[0,12,24,48,72,100], labels=[0,1,2,3,4])
df['TenureGroup'] = df['TenureGroup'].astype(int)

# Feature selection (using the same selected features from your notebook)
selected = ['tenure', 'PaperlessBilling', 'MonthlyCharges', 'SeniorCitizen', 
            'TenureGroup', 'TotalCharges', 'Partner', 'TotalServices', 'Dependents']

X = df[selected]
y = df['Churn']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)

# Save Logistic Regression model
joblib.dump(lr, 'lr_model.pkl')
print("✅ Logistic Regression model saved to 'lr_model.pkl'")