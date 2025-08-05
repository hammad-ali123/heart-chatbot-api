from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load your model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return "Heart Chatbot API is running"

# Prediction route for Dialogflow
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        params = data['queryResult']['parameters']

        # Extract values in the expected order
        user_input = [
            params['age'], params['sex'], params['cp'], params['trestbps'],
            params['chol'], params['fbs'], params['restecg'], params['thalach'],
            params['exang'], params['oldpeak'], params['slope'], params['ca'], params['thal']
        ]

        # Preprocess input
        input_array = scaler.transform([user_input])
        prediction = model.predict_proba(input_array)[0][1] * 100

        # Response to Dialogflow
        response_text = f"Your heart disease risk is {round(prediction, 2)}%. Please consider consulting a healthcare professional if you're concerned."

        return jsonify({"fulfillmentText": response_text})
    
    except Exception as e:
        return jsonify({"fulfillmentText": f"Something went wrong: {str(e)}"})
