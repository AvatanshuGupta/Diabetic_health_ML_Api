from pydantic import BaseModel,Field
from typing import Annotated,Literal
from enum import Enum
"""
<class 'pandas.core.frame.DataFrame'>
Index: 99982 entries, 0 to 99999
Data columns (total 9 columns):
 #   Column               Non-Null Count  Dtype  
---  ------               --------------  -----  
 0   gender               99982 non-null  float64
 1   age                  99982 non-null  float64
 2   hypertension         99982 non-null  int64  
 3   heart_disease        99982 non-null  int64  
 4   smoking_history      99982 non-null  float64
 5   bmi                  99982 non-null  float64
 6   HbA1c_level          99982 non-null  float64
 7   blood_glucose_level  99982 non-null  int64  
 8   diabetes             99982 non-null  int64  
dtypes: float64(5), int64(4)

"""
class Gender(Enum):
    FEMALE = 0.0
    MALE = 1.0

class SmokingHistory(Enum):
    NEVER = 0.0
    NO_INFO = 1.0
    EVER = 2.0
    FORMER = 3.0
    CURRENT = 4.0

class Patient(BaseModel):
    gender: Gender = Field(title="Gender of patient", description="0.0 for female, 1.0 for male")
    age: Annotated[float,Field(gt=0,lt=110,title="age of patient")]
    hypertension: Annotated[Literal[0,1],Field(title="If patient have hypertension",description="0 for no and 1 for yes")]
    heart_disease: Annotated[Literal[0,1],Field(title="If patient have any heart disease",description="0 for no and 1 for yes")]
    smoking_history: SmokingHistory = Field(title="Smoking History",description="Patient's smoking history")
    bmi: Annotated[float,Field(gt=8,lt=100,title="Body mass index",description="patients bmi")]
    HbA1c_level: Annotated[float,Field(gt=2,lt=20,title="patients hba1c",description="2 or 3 months average sugar profile")]
    blood_glucose_level: Annotated[int,Field(gt=20,lt=1000,title="blood glucose of patient",description="glucose level in mg/dl ")]

