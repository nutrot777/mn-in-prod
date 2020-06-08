from flask import Flask,render_template,request 
from imageio import imread
import cv2
import numpy as np 
from skimage.transform import rescale, resize
import keras.models
from keras.models import model_from_json
import re
import sys
import os
import base64
from keras.models import model_from_json
from keras.models import load_model
import tensorflow as tf
import json 
from keras.backend import set_session

global graph, model 
# Configuring the model to run in this environment 
config = tf.ConfigProto(
    device_count={'CPU': 1},
    intra_op_parallelism_threads = 1, 
    allow_soft_placement=True)

session = tf.Session(config=config)
keras.backend.set_session(session)




app = Flask(__name__)


json_file = open('model/model.json', 'r')
model_json = json_file.read()
json_file.close()

model = model_from_json(model_json)
model.load_weights('model/model.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
graph = tf.get_default_graph()

@app.route('/')
def index_view():
    return render_template('index.html')

def convertImage(imgData1):
    imgstr = re.search(b'base64,(.*)', imgData1).group(1)
    with open('output.png','wb') as output:
        output.write(base64.b64decode(imgstr))

@app.route('/predict', methods=[ 'GET','POST'])
def predict():
    imgData = request.get_data()
    convertImage(imgData)
    x= imread('output.png', pilmode='L')
    x = np.invert(x)
    x = resize(x, (28,28))
    x = x.reshape(1,28,28,1)
    
    with session.as_default():

        with session.graph.as_default():

            out = model.predict(x)
            
            response = np.array_str(np.argmax(out, axis=1))
            
            return response





if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')