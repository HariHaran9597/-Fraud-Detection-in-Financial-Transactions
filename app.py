# Step 1: Import necessary libraries
from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Step 2: Load the trained XGBoost model
# Replace 'xgboost_model.pkl' with the path to your saved model file
with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Step 3: Initialize the Flask app
app = Flask(__name__)

# Step 4: Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    """
    This route accepts POST requests with transaction data in JSON format.
    It returns a prediction (0 for non-fraudulent, 1 for fraudulent).
    """
    try:
        # Step 4.1: Get the input data from the request
        data = request.get_json()

        # Step 4.2: Convert the input data into a DataFrame
        # The input data should be a dictionary with keys matching the feature names
        input_data = pd.DataFrame([data])

        # Step 4.3: Make predictions using the XGBoost model
        prediction = model.predict(input_data)[0]  # Get the first prediction

        # Step 4.4: Return the prediction as a JSON response
        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        # Step 4.5: Handle errors gracefully
        return jsonify({'error': str(e)}), 400

# Step 5: Run the Flask app
if __name__ == '__main__':
    # Run the app on localhost (127.0.0.1) at port 5000
    app.run(debug=True)