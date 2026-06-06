import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import joblib

mlflow.set_experiment("Telco_Churn_Basic")

mlflow.sklearn.autolog()

df = pd.read_csv("telco_clean.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

with mlflow.start_run():

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

joblib.dump(
    model,
    "model.pkl"
)

mlflow.log_artifact(
    "model.pkl"
)