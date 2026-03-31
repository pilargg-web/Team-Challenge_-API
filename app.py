from flask import Flask, jsonify, request
import os
import joblib
import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.metrics.pairwise import haversine_distances
from sklearn.preprocessing import FunctionTransformer

# ---------------------------
# NECESARIO para cargar el .pkl
# ---------------------------
class BoxCoxTargetTransformer:
    def __init__(self):
        self.lambda_ = None

    def fit(self, y):
        y = np.array(y)
        _, self.lambda_ = stats.boxcox(y + 1e-6)
        return self

    def transform(self, y):
        return stats.boxcox(np.array(y) + 1e-6, lmbda=self.lambda_)

    def inverse_transform(self, y_bc):
        return inv_boxcox(y_bc, self.lambda_)

def add_distance(X):
    X = X.copy()
    coords = np.radians(X[['latitude', 'longitude']].values)
    center = np.radians([[40.4168, -3.7038]])
    X['distance_to_center_km'] = haversine_distances(coords, center) * 6371
    return X

# ---------------------------
# Cargar modelo
# ---------------------------
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open('pipeline_linear_regression_airbnb.pkl', 'rb') as f:
    model_data    = joblib.load(f)
    pipeline      = model_data["pipeline"]
    y_transformer = model_data["target_transformer"]

print("✅ Modelo cargado correctamente")

app = Flask(__name__)

# ---------------------------
# Landing page
# ---------------------------
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "servicio": "API de predicción de precios Airbnb Madrid 🏡",
        "endpoints": {
            "GET /":                "Esta página de bienvenida",
            "GET /api/v1/predict":  "Predicción de precio",
            "GET /api/v1/retrain":  "Reentrenar el modelo (extra voluntario)"
        },
        "parametros_predict": {
            "neighbourhood_group":              "string  (ej: Centro)",
            "neighbourhood":                    "string  (ej: Embajadores)",
            "room_type":                        "string  (ej: Entire home/apt)",
            "latitude":                         "float   (ej: 40.41)",
            "longitude":                        "float   (ej: -3.70)",
            "availability_365":                 "int     (ej: 200)",
            "calculated_host_listings_count":   "int     (ej: 1)",
            "reviews_per_month":                "float   (ej: 1.5)",
            "number_of_reviews":                "int     (ej: 30)",
            "minimum_nights":                   "int     (ej: 2)"
        },
        "ejemplo": (
            "/api/v1/predict?neighbourhood_group=Centro&neighbourhood=Embajadores"
            "&room_type=Entire home/apt&latitude=40.41&longitude=-3.70"
            "&availability_365=200&calculated_host_listings_count=1"
            "&reviews_per_month=1.5&number_of_reviews=30&minimum_nights=2"
        )
    })

# ---------------------------
# Predicción
# ---------------------------
@app.route('/api/v1/predict', methods=['GET'])
def predict():
    try:
        neighbourhood_group            = request.args.get('neighbourhood_group', type=str)
        neighbourhood                  = request.args.get('neighbourhood', type=str)
        room_type                      = request.args.get('room_type', type=str)
        latitude                       = request.args.get('latitude', type=float)
        longitude                      = request.args.get('longitude', type=float)
        availability_365               = request.args.get('availability_365', type=int)
        calculated_host_listings_count = request.args.get('calculated_host_listings_count', type=int)
        reviews_per_month              = request.args.get('reviews_per_month', type=float)
        number_of_reviews              = request.args.get('number_of_reviews', type=int)
        minimum_nights                 = request.args.get('minimum_nights', type=int)

        # Detectar parámetros faltantes
        params = {
            'neighbourhood_group':            neighbourhood_group,
            'neighbourhood':                  neighbourhood,
            'room_type':                      room_type,
            'latitude':                       latitude,
            'longitude':                      longitude,
            'availability_365':               availability_365,
            'calculated_host_listings_count': calculated_host_listings_count,
            'reviews_per_month':              reviews_per_month,
            'number_of_reviews':              number_of_reviews,
            'minimum_nights':                 minimum_nights
        }
        missing = [k for k, v in params.items() if v is None]
        if missing:
            return jsonify({"error": f"Faltan parámetros: {', '.join(missing)}"}), 400

        # Crear DataFrame y predecir
        input_data = pd.DataFrame([params])
        y_bc       = pipeline.predict(input_data)
        precio     = y_transformer.inverse_transform(y_bc)

        return jsonify({"precio_estimado_eur": round(float(precio[0]), 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------
# Retrain (extra voluntario)
# ---------------------------
@app.route('/api/v1/retrain', methods=['GET'])
def retrain():
    global pipeline, y_transformer
    try:
        if not os.path.exists("df_precios.csv"):
            return jsonify({"error": "No se encuentra df_precios.csv ❌"}), 404

        from sklearn.model_selection import train_test_split

        df = pd.read_csv("df_precios.csv")
        cols_to_drop = ["id","name","host_id","host_name","last_review","license"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        df = df.drop_duplicates().dropna()

        X = df.drop("price", axis=1)
        y = df["price"]
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        y_transformer.fit(y_train)
        y_train_bc = y_transformer.transform(y_train)
        pipeline.fit(X_train, y_train_bc)

        joblib.dump(
            {"pipeline": pipeline, "target_transformer": y_transformer},
            "pipeline_linear_regression_airbnb.pkl"
        )
        return jsonify({"status": "✅ Modelo reentrenado y guardado correctamente"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------
# Ejecutar
# ---------------------------
if __name__ == '__main__':
    app.run(debug=True)
