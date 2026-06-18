from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'model.joblib')
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.form.to_dict()
        
        # Convert values to correct types
        features = {
            'Age': int(data['Age']),
            'Diabetes': int(data['Diabetes']),
            'BloodPressureProblems': int(data['BloodPressureProblems']),
            'AnyTransplants': int(data['AnyTransplants']),
            'AnyChronicDiseases': int(data['AnyChronicDiseases']),
            'Height': int(data['Height']),
            'Weight': int(data['Weight']),
            'KnownAllergies': int(data['KnownAllergies']),
            'HistoryOfCancerInFamily': int(data['HistoryOfCancerInFamily']),
            'NumberOfMajorSurgeries': int(data['NumberOfMajorSurgeries'])
        }
        
        # Calculate BMI (required by the new model)
        height_m = features['Height'] / 100
        features['BMI'] = features['Weight'] / (height_m ** 2)
        
        # Add the extra features the latest model was trained on
        features['Age2'] = features['Age'] ** 2
        features['BMI2'] = features['BMI'] ** 2
        features['Age_BMI'] = features['Age'] * features['BMI']
        features['Total_Diseases'] = features['Diabetes'] + features['BloodPressureProblems'] + features['AnyTransplants'] + features['AnyChronicDiseases']
        features['Age_Diseases'] = features['Age'] * features['Total_Diseases']
        
        # Create DataFrame for prediction
        input_df = pd.DataFrame([features])
        
        # Reorder columns to match training EXACTLY
        try:
            model_cols = joblib.load('model_columns.joblib')
            input_df = input_df[model_cols]
        except:
            pass
        
        # Predict
        prediction = model.predict(input_df)[0]
        
        return jsonify({'prediction': f"₹{prediction:,.2f}"})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
