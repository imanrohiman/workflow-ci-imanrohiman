import joblib
import pandas as pd

model = joblib.load("../Membangun_model/model.pkl")

sample = pd.DataFrame({
    "Pclass":[3],
    "Sex":[0],
    "Age":[22],
    "SibSp":[1],
    "Parch":[0],
    "Fare":[7.25]
})

pred = model.predict(sample)

print("Prediction:", pred)
