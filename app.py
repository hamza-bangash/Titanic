from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
# Load the pre-trained model
with open('src/model.pkl', 'rb') as f:
    model =pickle.load(f)
# Load the column types and categories
with open('src/col_types.pkl', 'rb') as f:
    col_types = pickle.load(f)
# Load the categories for categorical features
with open('src/categories.pkl', 'rb') as f:
    categories = pickle.load(f)
# Define the feature names based on the column types
FEATURE_NAMES = list(col_types.keys())

@app.route('/')
def index():
    return  render_template(
        'index.html',
        col_types=col_types,
        categories=categories
    )



@app.route('/predict', methods=['POST'])
def predict():
    data = {}
    for col in FEATURE_NAMES:
        val = request.form[col]
        dtype = col_types[col]
        if 'float' in dtype:
            data[col] = float(val)
        elif 'int' in dtype:
            data[col] = int(val)
        else:
            data[col] = val
    df = pd.DataFrame([data], columns=FEATURE_NAMES)
    pred = model.predict(df)[0]
    if pred == 0:
        return render_template('result.html', survive='NO')
    else:
        return render_template('result.html', survive='YES')

    

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return jsonify({"prediction":(prediction)})

if __name__ == "__main__":
    app.run(debug=True)