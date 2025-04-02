from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("titanic_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert input data to numpy array
        features = np.array(data["features"]).reshape(1, -1)  # Ensure correct shape

        # Make prediction
        prediction = model.predict(features)[0]  # Get first prediction
        
        # Return JSON response
        return jsonify({"survived": int(prediction)})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
