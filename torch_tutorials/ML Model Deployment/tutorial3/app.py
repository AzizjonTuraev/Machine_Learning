# Youtube: https://www.youtube.com/watch?v=b5F667g1yCk

import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pickle
import pandas as pd



app = FastAPI()
classifier = pickle.load(open("classifier.pkl", "rb"))

@app.get("/")
def index():
    return {"message": "Hello, stranger"}

@app.get("/Welcome")
def get_name(name:str):
    return {"Welcome to Krish Youtube Channel": f"{name}"}

@app.post("/predict")
def predict_species(data:BankNote):
    data = data.dict()
    print(data)
    print("Hello")
    variance = data["variance"]
    skewness = data["skewness"]
    curtosis = data["curtosis"]
    entropy = data["entropy"]
    print(classifier.predict([[variance, skewness, curtosis, entropy]]))
    print("Hello")
    pred = classifier.predict([[variance, skewness, curtosis, entropy]])
    if pred[0]>0.5:
        prediction = "Fake note"
    else:
        prediction = "Real note"
    return {"prediction" : prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

# on command shell
# uvicorn app:app --reload