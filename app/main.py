from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from src.model import ServiceProviderClassifier
from src.data_preprocessing import extract_phone_prefixes

main_dir = os.path.dirname(__file__)
model_path = os.path.join(main_dir, '../model/service_provider_classifier.joblib')
model_path = os.path.abspath(model_path)

app = FastAPI()
classifier = ServiceProviderClassifier()
classifier.load_model(model_path)

class TelephoneNumber(BaseModel):
    Telephone_Number: str

@app.post("/predict")
async def predict_service_provider(telephone_number: TelephoneNumber):
    try:
        data = pd.DataFrame({'Telephone_Number': [telephone_number.Telephone_Number]})
        preprocessed_data = extract_phone_prefixes(data)
        prediction = classifier.predict(preprocessed_data)
        return {"ServiceProvider": prediction[0],"ResponseCode":"00","ResponseDescription":"Successful"}
    
    except ValueError as e:
        #raise HTTPException(status_code=500, detail=str(e))
        print(type(e))
        return {"ServiceProvider": "","ResponseCode":"04","ResponseDescription":"Unknown Network Provider"}

    except Exception as e:
        #raise HTTPException(status_code=500, detail=str(e))
        print(type(e))
        return {"ServiceProvider": "","ResponseCode":"01","ResponseDescription":"Something went wrong!"}

        
    