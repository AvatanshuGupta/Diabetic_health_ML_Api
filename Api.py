from fastapi import FastAPI
from src.DataSchema.schema import Patient
from src.pipeline.prediction_pipeline import Prediction
from src.logger import logging
import sys
from src.exception import CustomException
app=FastAPI()
@app.get('/')
def welcome():
    return {'message':'Welocome to diabetic health prediction model api.'}

@app.get('/predict')
def predict(patient:Patient):
    input_data={
        'gender':patient.gender,
        'age':patient.age,
        'hypertension':patient.hypertension,
        'heart_disease':patient.heart_disease,
        'smoking_history':patient.smoking_history,
        'bmi ':patient.bmi,
        'HbA1c_level':patient.HbA1c_level,
        'blood_glucose_level':patient.blood_glucose_level
    }
    try:
        pred=Prediction(input_data=input_data)
        logging.info("prediction done successfully in api")
        return {"prediction":pred[0]}
    except Exception as e:
        raise CustomException(e,sys)
