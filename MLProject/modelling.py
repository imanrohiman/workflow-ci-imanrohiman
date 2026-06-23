import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import joblib
import warnings

warnings.filterwarnings('ignore')

# ★★★ WAJIB: Aktifkan MLflow autolog ★★★
mlflow.autolog()  # ← TAMBAHKAN INI!

def load_and_preprocess_data():
    df = pd.read_csv('../namadataset_preprocessing/titanic.csv')
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    le = LabelEncoder()
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col].astype(str))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Manual logging
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"✅ Model accuracy: {accuracy:.4f}")
        print(f"📁 Run ID: {run.info.run_id}")
        
        joblib.dump(model, 'model.pkl')
        print("✅ Model saved to model.pkl")
        
        return model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    model = train_model(X_train, y_train, X_test, y_test)
