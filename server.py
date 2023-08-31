# Load libraries
import flask
from flask import request
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.python.keras.models import load_model

# instantiate flask 
app = flask.Flask(__name__)

model_path = "D:/neural networks/DeepAudioProjects/Capuchin Calls/maybefinal.h5"
model = load_model(model_path)

@app.route("/model",methods=["POST"])
def serve_model():
    request_data = request.get_json(force=True)
    # request_data
    file = request_data["audio_file"]
    return ("Prediction is {}".format(request_data["class"]))

if __name__ == "__main__":
    app.run()
