from flask import Flask,render_template,request
from keras.models import load_model
import cv2
import numpy as np
import os


app = Flask(__name__)

# Home page
@app.route("/",methods=['GET','POST'])
def hello_world():
    pass
    return render_template('index.html')

# Single image page
@app.route("/predict_1",methods=['GET','POST'])
def predict_1():
    pass
    return render_template('predict_1.html')

# Multiple Image Page
@app.route("/predict_mul",methods=['GET','POST'])
def predict_mul():
    pass
    return render_template('predict_mul.html')


if __name__ == "__main__":
    app.debug = True
    app.run(port=9899,debug=False)