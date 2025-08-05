from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialise Flask app
app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Home route (used for health check)
@app.route("/")
def home():
    return "Heart Chatbot API is running"

# Prediction route for Dialogflow
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure request is JSON
        if not request.is_json:
            return jsonify({"fulfillmentText": "Request content-type must be JSON"}), 400

        # Parse JSON from Dialogflow
        data = request.get_json(force=True)

        # Extract parameters safely
        if "queryResult" not in data or "parameters" not in data["queryResult"]:
            return jsonify({"fulfillmentText": "Missing or malformed parameters."}), 400

        params = data["queryResult"]["parameters"]

        # Validate all required fields
        required_keys = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]

        for key in required_keys:
            if key not in params:
                return jsonify({"fulfillmentText": f"Missing parameter: {key}"}), 400

        # Prepare model input
        input_data = [
            params["age"],
            params["sex"],
            params["cp"],
            params["trestbps"],
            params["chol"],
            params["fbs"],
            params["restecg"],
            params["thalach"],
            params["exang"],
            params["oldpeak"],
            params["slope"],
            params["ca"],
            params["thal"]
        ]

        input_array = scaler.transform([input_data])
        prediction = model.predict_proba(input_array)[0][1] * 100

        # Response for Dialogflow
        response_text = f"Your heart disease risk is {round(prediction, 2)}%. Please consult a doctor if concerned."
        return jsonify({"fulfillmentText": response_text})

    except Exception as e:
        return jsonify({"fulfillmentText": f"Server error: {str(e)}"}), 500

# Required for Render.com deployment
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
