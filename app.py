from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Home route
@app.route('/')
def home():
    return "Heart Chatbot API is running"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        params = data['queryResult']['parameters']

        input_data = [
            params['age'], params['sex'], params['cp'], params['trestbps'],
            params['chol'], params['fbs'], params['restecg'], params['thalach'],
            params['exang'], params['oldpeak'], params['slope'], params['ca'], params['thal']
        ]

        input_array = scaler.transform([input_data])
        prediction = model.predict_proba(input_array)[0][1] * 100

        response_text = f"Your heart disease risk is {round(prediction, 2)}%. Please consult a doctor if concerned."

        return jsonify({"fulfillmentText": response_text})

    except Exception as e:
        return jsonify({"fulfillmentText": f"Error occurred: {str(e)}"})
