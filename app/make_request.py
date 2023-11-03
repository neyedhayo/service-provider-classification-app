import requests

url = "http://127.0.0.1:8000/predict"
data = {"Telephone_Number": "08109475645"}
response = requests.post(url, json=data)
print(response.json())

