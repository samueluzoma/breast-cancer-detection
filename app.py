from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained artifacts
# Ensure cancer_model.pkl and scaler.pkl are in the same folder as this script
try:
    model = joblib.load('cancer_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model files: {e}")

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Breast Cancer Diagnostic API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get JSON data from request
        data = request.get_json()
        
        # 2. Extract features (expecting a list of 30 values)
        # Ensure the order matches the original training columns
        features = np.array(data['features']).reshape(1, -1)
        
        # 3. Apply the same scaling used during training
        scaled_features = scaler.transform(features)
        
        # 4. Perform prediction
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0]
        
        # 5. Map 0/1 back to Benign/Malignant
        result = "Malignant" if prediction == 1 else "Benign"
        confidence = float(max(probability))
        
        return jsonify({
            'status': 'success',
            'prediction': result,
            'confidence': f"{confidence:.2%}",
            'class_id': int(prediction)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

# if __name__ == '__main__':
#     # Local development port
#     port = int(os.environ.get("PORT", 5002))
#     app.run(host='0.0.0.0', port=port, debug=True)
if __name__ == '__main__':
    # Render provides a $PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)