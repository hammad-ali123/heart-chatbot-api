from flask import Flask, request, jsonify
import numpy as np
import joblib
import shap
import xgboost as xgb

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
explainer = shap.Explainer(model)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json['data']
    input_array = scaler.transform([input_data])
    prediction = model.predict_proba(input_array)[0][1]

    shap_value = explainer(input_array)
    feature_impact = shap_value.values[0].tolist()

    return jsonify({
        'risk_score': round(prediction * 100, 2),
        'explanation': feature_impact
    })

if __name__ == '__main__':
    app.run(port=5000)
