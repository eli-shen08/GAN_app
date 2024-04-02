from flask import Flask,render_template,request,send_file
from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot
import numpy as np

app = Flask(__name__)


model = load_model('generator_model_180.0.h5')


# Home page
@app.route("/",methods=['GET','POST'])
def hello_world():
    pass
    return render_template('index.html')



# Function to generate noise which returns EX: 64, 32, 32, 3
def generate_noise(noise,n_samples):
    x_input = randn(noise*n_samples)
    x_input = x_input.reshape(n_samples,noise)
    return x_input


# Function to create plot from model prediction .
# def create_plot(examples,n):
#     for i in range(n * n):
#         pyplot.subplot(n, n, 1+i)
#         pyplot.axis('off')
#         pyplot.imshow(examples[i])
#     pyplot.show()

# Single image page
@app.route("/predict_1",methods=['GET','POST'])
def predict_1():
    print(request.method)
    latent_points = generate_noise(100,49)
    X = model.predict(latent_points)
    X = (X + 1) / 2
    # X = create_plot(X,1)
    Y = X[10]
    pyplot.imsave('single_predicted_image.png', Y)
    # random_number = np.random.randint(12)
    return send_file('single_predicted_image.png', mimetype='image/png')

# Multiple Image Page
@app.route("/predict_mul",methods=['GET','POST'])
def predict_mul():
    pass
    return render_template('predict_mul.html')


if __name__ == "__main__":
    app.debug = True
    app.run(port=9899,debug=True)