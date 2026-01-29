#I am using Flask for my backend
#Blueprint allows me to create modular components for my Flask application
#Request and jsonify are used to handle incoming requests and send JSON responses from the frontend
#Pickle is used to load the pre-trained machine learning model, as can be found in the 'model' folder
#OS is used to handle file paths, making the code more clean
#Numpy is used to create an array for the model input
from flask import Flask, Blueprint, request, jsonify
import pickle
import os
import numpy as np

#This initializes a Blueprint for the score API (creates the modular component)
score_api = Blueprint('score_api', __name__)

# Load model
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'exam_score_model.pkl')
#RB means read binary, which is necessary for loading pickle files as they are stored in binary
#with means that the file will be properly closed the code block is finished, which is good for resource management
#the model_file just makes the code a bit more readable and allows me to use it in the next line again nicely
print(f"Model path: {model_path}")
print(f"Model file exists: {os.path.exists(model_path)}")
try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    print(f"ERROR: Model file not found at {model_path}")
    model = None
except Exception as e:
    print(f"ERROR loading model: {e}")
    model = None

#This defines a route for the score API that listens for POST requests at the /score endpoint
@score_api.route("/score", methods=['GET', 'POST']) #specifies that this route only accepts POST requests and not GET requests
def predict(): #What will happen when a request is made to this endpoint
    try:
        if model is None:
            return jsonify({'error': 'Model failed to load. Please run v-two.py first to generate exam_score_model.pkl'}), 400
            
        data = request.json #gets the data from what the frontend sent
        # Extracts features from request in the correct order matching training script (v-two.py)
        features = np.array([[data['study_hours'], data['age'], data['sleep_hours'], data['class_attendance'], data['exam_difficulty'], data['sleep_quality'], data['facility_rating']]])
        
        # Makes prediction using the pipeline (which includes both scaler and model, so I don't have to do scaling here)
        prediction = model.predict(features)
        # Clips prediction to be between 0 and 100 (exam scores can't be outside this range)
        prediction_clipped = np.clip(prediction[0], 0, 100)
        
        #returns the prediction as a JSON response, ready for the frontend to use
        return jsonify({
            'predicted_score': float(prediction_clipped[0])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400