from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the models
reg_model = joblib.load('fish_weight_predictor.pkl')
clf_model = joblib.load('fish_species_classifier.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_weight', methods=['POST'])
def predict_weight():
    # Extracting input feature values
    input_features = [float(x) for x in request.form.values()]
    features = [np.array(input_features)]
    
    # Predicting
    prediction = reg_model.predict(features)
    
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text=f'Estimated Fish Weight: {output} grams')

@app.route('/predict_species', methods=['POST'])
def predict_species():
    # Extracting input feature values
    input_features = [float(x) for x in request.form.values()]
    features = [np.array(input_features)]
    
    # Predicting
    prediction = clf_model.predict(features)
    
    species = prediction[0]
    
    return render_template('index.html', prediction_text=f'Predicted Fish Species: {species}')

if __name__ == "__main__":
    app.run(debug=True)
