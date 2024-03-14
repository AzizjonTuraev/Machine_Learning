# Youtube : https://www.youtube.com/watch?v=ZTz26f6XXrQ

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json

app = FastAPI()

class model_input(BaseModel):

    Pregnancies : int
    Glucose : int
    BloodPressure : int
    SkinThickness : int
    Insulin : int
    BMI : float
    DiabetesPedigreeFunction: float
    Age : int


diabetes_model = pickle.load(open("diabetes_model.sav", "rb"))

@app.post("/diabetes_prediction")
def diabetes_pred(input_params: model_input):

    input_data = input_params.json()
    input_dict = json.loads(input_data)

    preg = input_dict["Pregnancies"]
    glu = input_dict["Glucose"]
    bp = input_dict["BloodPressure"]
    skin = input_dict["SkinThickness"]
    insulin = input_dict["Insulin"]
    bmi = input_dict["BMI"]
    dpf = input_dict["DiabetesPedigreeFunction"]
    age = input_dict["Age"]

    input_list = [preg, glu, bp, skin, insulin, bmi, dpf, age]
    prediction = diabetes_model.predict([input_list])

    if prediction[0]==0:
        return "The person is not diabetic"
    else:
        return "The person is diabetic"
