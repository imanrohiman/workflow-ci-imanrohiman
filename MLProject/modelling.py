# modelling.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import joblib
import os

def main():
    # Set tracking URI
    # mlflow.set_tracking_uri("http://localhost:5000")
    
    with mlflow.start_run() as run:
        # Load preprocessed data
        # data_path = "namadataset_preprocessing/data_clean.csv"
	data_path = os.path.join(os.path.dirname(__file__), "../namadataset_preprocessing/data_clean.csv")
        df = pd.read_csv(data_path)
        
        # Split features and target (sesuaikan dengan dataset Anda)
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split train test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Log parameters
        n_estimators = 100
        max_depth = 10
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        os.makedirs("model", exist_ok=True)
        joblib.dump(model, "model/model.pkl")
        
        print(f"Model accuracy: {accuracy}")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
