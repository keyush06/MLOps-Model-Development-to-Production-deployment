import predictions
import warnings
import requests
warnings.filterwarnings("ignore") 

ride = {
    "PULocationID": 1,
    "DOLocationID": 35,
    "trip_distance": 145
}

url = "http://127.0.0.1:9090/predict"
response = requests.post(url, json = ride)
print(response.json())

## Just for testing
# features = predictions.prepare_features(ride)
# print("Features:", features)
# preds = predictions.predict(features)
# print('preds', preds)