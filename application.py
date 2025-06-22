#Import required libraries

from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Define feature names in the order the model expects
feature_names = ['qty', 'total_price', 'freight_price', 'product_name_lenght',
                 'product_description_lenght', 'product_photos_qty', 'product_weight_g',
                 'product_score', 'customers', 'weekday', 'weekend', 'holiday',
                 'month', 'year', 's', 'volume', 'comp_1', 'ps1', 'fp1',
                 'comp_2', 'ps2', 'fp2', 'comp_3', 'ps3', 'fp3', 'lag_price']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract inputs from form
        input_data = [float(request.form[feature]) for feature in feature_names]

        # Convert to array and reshape
        features = np.array(input_data).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]

        return render_template('index.html', prediction=f"Predicted Unit Price: ${prediction:.2f}")

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

