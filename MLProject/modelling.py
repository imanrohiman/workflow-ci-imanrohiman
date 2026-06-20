import pandas as pd
import numpy as np
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import mlflow

def preprocess_data(df):
    """Preprocess dataset before training"""
    # Create a copy
    df = df.copy()
    
    # Drop unnecessary columns (if they exist)
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # Handle missing values
    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].median())
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    if 'Fare' in df.columns:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # Encode categorical columns
    if 'Sex' in df.columns:
        df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    if 'Embarked' in df.columns:
        df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    
    # Scale numerical columns
    scaler = StandardScaler()
    numeric_cols = ['Age', 'Fare']
    existing_numeric = [col for col in numeric_cols if col in df.columns]
    if existing_numeric:
        df[existing_numeric] = scaler.fit_transform(df[existing_numeric])
    
    return df

def train_model(X_train, y_train):
    """Train Random Forest model"""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True)
    args = parser.parse_args()
    
    print("Loading data from:", args.data_file)
    df = pd.read_csv(args.data_file)
    
    print("Original data shape:", df.shape)
    
    # Preprocess data
    print("Preprocessing data...")
    df_processed = preprocess_data(df)
    print("Processed data shape:", df_processed.shape)
    
    # Split features and target
    X = df_processed.drop('Survived', axis=1)
    y = df_processed['Survived']
    
    print("Training model...")
    model = train_model(X, y)
    
    # Log to MLflow
    with mlflow.start_run():
        accuracy = accuracy_score(y, model.predict(X))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "random_forest_model")
        print("Model trained and logged to MLflow")
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Done!")
