from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model and scaler
crop_model = joblib.load('rff_crop_model.pkl')      # RandomForestClassifier for crop prediction
yield_model = joblib.load('rff_yield_model.pkl')    # RandomForestRegressor for yield prediction
scaler = joblib.load('scaler_model.pkl')                  # Scaler for feature normalization

# Columns in the training dataset
X_train_encoded_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Crop labels (same as your crop dataset labels)
crop_labels = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans',
               'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango',
               'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut',
               'cotton', 'jute', 'coffee']

# Function to predict both crop and yield
def predict_crop_and_yield(input_data):
    # Scale input data (remove column names to avoid the warning)
    input_scaled = scaler.transform(input_data[X_train_encoded_columns])  # Properly select columns
    # Predict crop (classification)
    predicted_crop = crop_model.predict(input_scaled)[0]  # Get the predicted crop (it may be the name or index)

    # If the predicted crop is an index, map it to the crop name
    if isinstance(predicted_crop, int):
        predicted_crop = crop_labels[predicted_crop]  # Map index to crop name

    # Predict yield (regression)
    predicted_yield = yield_model.predict(input_scaled)[0]  # Predict yield in tons

    return predicted_crop, predicted_yield

# Main route to handle user inputs and predictions
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = request.get_json()

        # Extract input data from the request (convert them to float)
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])

        # Create a DataFrame for input data
        input_data = pd.DataFrame({
            'N': [N],
            'P': [P],
            'K': [K],
            'temperature': [temperature],
            'humidity': [humidity],
            'ph': [ph],
            'rainfall': [rainfall],
        })

        # Get predictions for crop and yield
        predicted_crop, predicted_yield = predict_crop_and_yield(input_data)

        # Return the predictions as JSON response
        return jsonify({
            'predicted_crop': predicted_crop,
            'predicted_yield': round(predicted_yield, 2)  # Round yield to two decimal places
        })

    return render_template('index.html', predicted_crop=None, predicted_yield=None)

@app.route('/learn')
def learn():
    return render_template('learn.html')

if __name__ == '__main__':
    app.run(debug=True)
