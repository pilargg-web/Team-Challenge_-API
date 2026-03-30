from fastapi import FastAPI
import pickle

app = FastAPI()

# Cargar modelo
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"mensaje": "API funcionando 🚀"}

@app.post("/predict")
def predict(data: dict):
    features = list(data.values())
    prediction = model.predict([features])
    return {"prediccion": int(prediction[0])}