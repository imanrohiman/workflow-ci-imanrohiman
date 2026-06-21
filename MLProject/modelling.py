import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
import joblib
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='namadataset_preprocessing/data_clean.csv')
    args = parser.parse_args()
    
    with mlflow.start_run() as run:
        # Load preprocessed data
        data_path = os.path.join(os.path.dirname(__file__), "../namadataset_preprocessing/data_clean.csv")
        df = pd.read_csv(data_path)
        
        print("📊 Data shape:", df.shape)
        print("📋 Columns:", df.columns.tolist())
        print("🔍 Data types:\n", df.dtypes)
        
        # Pisahkan fitur dan target (kolom terakhir = target)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # 🔧 ENCODE STRING KE ANGKA
        print("\n🔄 Encoding categorical columns...")
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                print(f"✅ Column '{col}' encoded")
        
        # Encode target jika string
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            print(f"✅ Target column encoded")
        
        print("\n📊 X shape:", X.shape)
        print("📊 y shape:", y.shape)
        
        # Split train-test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Training model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log ke MLflow
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
        
        # Print hasil
        print(f"\n✅ Model accuracy: {accuracy:.4f}")
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        os.makedirs("model", exist_ok=True)
        joblib.dump(model, "model/model.pkl")
        print("✅ Model saved to model/model.pkl")

if __name__ == "__main__":
    main()
