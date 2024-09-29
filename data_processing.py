import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

    label_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                  'InternetService', 'OnlineSecurity', 'DeviceProtection', 'TechSupport', 
                  'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

    for col in label_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    scaler = StandardScaler()
    df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])
    
    return df
