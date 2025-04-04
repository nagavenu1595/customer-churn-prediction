import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

def preprocess_data():
    # Load dataset
    df = pd.read_csv("../data/raw/churn.csv")
    
    # Clean column names (replace spaces with underscores)
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    
    # Handle TotalCharges conversion (ensure it's numeric)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Convert Churn target from 'Yes'/'No' to 1/0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)
    
    # Drop non-feature column (customerID)
    df.drop(columns=['customerID'], inplace=True, errors='ignore')
    
    # Convert categorical variables (only for columns that are actually categorical)
    # Define the categorical columns explicitly to avoid encoding numerical ones like TotalCharges
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                        'MultipleLines', 'InternetService', 'OnlineSecurity', 
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                        'StreamingTV', 'StreamingMovies', 'Contract', 
                        'PaperlessBilling', 'PaymentMethod']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Split features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Train-test split (with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Scale numerical features (only scaling features that are numeric)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create directories if they don't exist
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Save processed data
    pd.DataFrame(X_train, columns=X.columns).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_test, columns=X.columns).to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("✅ Data preprocessing complete!")
    # Now run feature selection
    select_features()

def select_features():
    # Load processed data
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    
    # Apply feature selection: select top 10 features using ANOVA F-test
    selector = SelectKBest(score_func=f_classif, k=10)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Retrieve names of selected features
    selected_features = X_train.columns[selector.get_support()]
    
    # Save selected features data
    pd.DataFrame(X_train_selected, columns=selected_features).to_csv("data/processed/X_train_selected.csv", index=False)
    pd.DataFrame(X_test_selected, columns=selected_features).to_csv("data/processed/X_test_selected.csv", index=False)
    joblib.dump(selector, 'models/feature_selector.pkl')
    
    print(f"✅ Selected features: {list(selected_features)}")

if __name__ == "__main__":
    preprocess_data()
