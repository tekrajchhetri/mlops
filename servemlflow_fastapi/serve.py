from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from requests.exceptions import ConnectionError
import pickle
import numpy as np
import pandas as pd
from io import BytesIO
from io import StringIO
import requests

app = FastAPI()

class HousingPrice(BaseModel):
    latitude: float
    longitude:float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    median_house_value: float
    ocean_proximity_nearby: float
    ocean_proximity_less1h_ocean: float
    ocean_proximity_inland: float
    ocean_proximity_near_ocean: float
    ocean_proximityisland: float
    ocean_proximity_nearocean: float


@app.post('/predict')
async def predict_house_price(house: HousingPrice):
    data = house.dict()
    data_for_pred = [
        [
            data['latitude'],
            data['longitude'],
            data['housing_median_age'],
            data['total_rooms'],
            data['total_bedrooms'],
            data['population'],
            data['households'],
            data['median_income'],
            data['ocean_proximity_less1h_ocean'],
            data['ocean_proximity_inland'],
            data['ocean_proximityisland'],
            data['ocean_proximity_nearby'], #nearbay
            data['ocean_proximity_nearocean'],



        ]
    ]

    api_endpoint = "http://localhost:5008/invocations"
    inference_req = {
        "dataframe_records": data_for_pred
    }
    response = requests.post(api_endpoint, json=inference_req)
    return  {
        "prediction": response.text,
    }

def clean_n_process(df):
    df = df.dropna()
    categorical_columns = list(df.select_dtypes(exclude="number").columns)
    df = pd.get_dummies(df, columns=categorical_columns, dtype=int)
    return df

@app.post('/files/')
async def predict_batch_files(file: UploadFile = File(...)):
    contents = file.file.read()
    data = BytesIO(contents)
    df = pd.read_csv(data)
    dfprocessed = clean_n_process(df)
    data.close()
    file.file.close()

    try:
        conn_working = requests.get("http://127.0.0.1:5000/")
        conn_working = True if conn_working.status_code == 200 else False
    except:
        print("error")
        conn_working = False

    if conn_working:
        lst = dfprocessed.values.tolist()
        api_endpoint = "http://localhost:5008/invocations"

        inference_req = {
            "dataframe_records": lst
        }
        response = requests.post(api_endpoint, json=inference_req)
        return {
            "prediction": response.text,
        }
    else:
        # mlflow is not running load local model
        print("Model loaded")
        loadmodel = pickle.load(open("model.pkl","rb"))
        prediction = loadmodel.predict(dfprocessed)
        return {
            "prediction": list(prediction),
        }




@app.get("/")
async def root():
    return {"message": "House price prediction service"}