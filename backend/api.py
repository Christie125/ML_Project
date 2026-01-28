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
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

#This defines a route for the score API that listens for POST requests at the /score endpoint
@score_api.route("/score", methods=['GET', 'POST']) #specifies that this route only accepts POST requests and not GET requests
def predict(): #What will happen when a request is made to this endpoint
    try:
        data = request.json #gets the data from what the frontend sent
        # Extracts features from request, ready to go into the model input
        features = np.array([data['age'], data['study_hours'], data['sleep_hours'], data['class_attendance'], data['exam_difficulty'], data['sleep_quality'], data['facility_rating']]).reshape(1, -1)
        
        # Makes prediction
        prediction = model.predict(features)
        
        #returns the prediction as a JSON response, ready for the frontend to use
        print(prediction)
        return jsonify({
            'predicted_score': float(prediction[0])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400