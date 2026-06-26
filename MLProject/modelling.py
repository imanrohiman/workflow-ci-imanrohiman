import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import joblib
import warnings

warnings.filterwarnings("ignore")

# Aktifkan MLflow autolog
mlflow.sklearn.autolog()


def load_and_preprocess_data():
    # Load dataset HASIL preprocessing
    df = pd.read_csv("../preprocessing/namadataset_preprocessing/titanic.csv")

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
    )

    with mlflow.start_run():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")

        # Simpan model
        joblib.dump(model, "model.pkl")

    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    model = train_model(X_train, y_train, X_test, y_test)