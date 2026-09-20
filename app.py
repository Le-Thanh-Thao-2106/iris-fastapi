from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

model=joblib.load("svm_model.pkl")
iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="macro"
)

recall = recall_score(
    y_test,
    y_pred,
    average="macro"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)
app=FastAPI(
    title="IrisClassification API",
    description="SVM modelforthe Irisdataset",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class IrisInput(BaseModel):
    sepal_length:float
    sepal_width:float
    petal_length:float
    petal_width:float
species={
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}
@app.get("/")
def home():
    return {"message":"IrisSVMAPI isrunning"}
@app.get("/health")
def health():
    return {"status": "healthy"}
@app.get("/metrics")
def metrics():
    return {
        "model": "Linear SVM",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
@app.post("/predict")
def predict(data:IrisInput):
    features =[[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]]
    prediction=int(model.predict(features)[0])
    return {
        "class_id":prediction,
        "prediction":species[prediction],
    }