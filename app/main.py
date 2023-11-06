from fastapi import FastAPI, HTTPException, status, Request, Depends
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

# generating access to the app by user-authentication
API_KEY = os.getenv("MY_SECRET_API_KEY")
def get_api_key(request: Request):
    api_key = request.headers.get("Authorization")
    
    if api_key == f"Bearer {API_KEY}":
        return api_key

    else:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Could not validate credentials",
        )

@app.get("/private-area/")
def read_private_area(api_key: str = Depends(get_api_key)):
    return {"message": "Hello, private user!"}

# base model telephone number class
class TelephoneNumber(BaseModel):
    Telephone_Number: str

# POST request to test the model, from preprocessing steps to prediction
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

        
    