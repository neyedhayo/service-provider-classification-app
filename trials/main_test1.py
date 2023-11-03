from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from trial_directory.model_test1 import load_model, load_label_encoder, load_preprocessing
from src.data_preprocessing import prefix_extractor, preprocessor #, pipeline


app = FastAPI()
model = load_model()
label_encoder = load_label_encoder()
preprocessed = load_preprocessing()

class Item(BaseModel):
    Telephone_Number: str


@app.post("/predict")
# async def predict(item: Item):
#     try:
#         # converting the input data into a DataFrame
#         data = pd.DataFrame({"Telephone_Number": [item.Telephone_Number]})

#         # preprocess the data
#         data = preprocessor.fit_transform(data)    

#         # getting the prediction
#         prediction = model.predict(data) # await

#         # converting the prediction into the original label
#         label = label_encoder.inverse_transform(prediction)
#         # label = label_encoder.classes_[prediction[0]]

#         # return statement
#         return {"prediction": label[0]}
#         # return {"prediction": label}
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

async def predict(item: Item):
    # converting the input data into a DataFrame
    data = pd.DataFrame({"Telephone_Number": [item.Telephone_Number]})

    # preprocess the data
    # transformed_data = pipeline.fit_transform(data)    

    # getting the prediction
    # prediction = model.predict(transformed_data) # await
    prediction = model.predict(preprocessed)

    # converting the prediction into the original label
    label = label_encoder.inverse_transform(prediction)
    # label = label_encoder.classes_[prediction[0]]

    # return statement
    return {"prediction": label[0]}
    # return {"prediction": label}

