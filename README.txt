# Heart Disease Risk Chatbot Backend

## Setup Instructions

1. Install dependencies:
   pip install -r requirements.txt

2. Train your model and save it as `model.pkl` and `scaler.pkl`.

3. Start the Flask server:
   python app.py

4. Use ngrok to expose the endpoint:
   ngrok http 5000

5. Set the ngrok URL as your webhook in Dialogflow.
