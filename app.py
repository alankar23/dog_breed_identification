from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
# Flask
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

#keras 
from keras.models import load_model

# Flask app
app = Flask(__name__)

# Model path 
PATH = 'dog_breed_model.h5'

# Load model
model = load_model(PATH)
model.make_predict_function()
print('Model Loaded. Check Localhost')


from numpy import load

labels = load('labels.npy')
label_encoder = LabelEncoder()
vec = label_encoder.fit_transform(labels)





def model_predict(img_path,model):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (300, 300))
    image = image.reshape(1,300,300,3)
    pred = model.predict(image)
    return pred

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        filex =request.files['file']
        base = os.path.dirname(__file__)
        file_path = os.path.join(base, 'uploads', secure_filename(filex.filename))
        filex.save(file_path)

        preds = model_predict(file_path,model)
        preds = preds.argmax(axis=-1)
        preds = label_encoder.inverse_transform(preds)
        return str(preds[0])
    return None


if __name__ == '__main__':
    app.run(debug=True)

