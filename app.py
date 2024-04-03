from flask import Flask,render_template,request,send_file
from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot
from PIL import Image
import numpy as np
import cv2
import os

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
    # picking a random img index to output
    rand_idx = np.random.randint(0,49)
    Y = X[rand_idx]
    print(X[rand_idx].shape)
    # Upscale the image by a factor of 2 using nearest neighbor interpolation
    Y  = cv2.resize(Y, (256, 256), interpolation=cv2.INTER_NEAREST)
    pyplot.imsave('single_predicted_image.png', Y, dpi=70)
    return send_file('single_predicted_image.png', mimetype='image/png')







# Multiple Image Page
@app.route("/predict_mul",methods=['GET','POST'])
def predict_mul():
    imgs_path = []
    print(request.method)
    latent_points = generate_noise(100,49)
    X = model.predict(latent_points)
    X = (X + 1) / 2
    X = ((X + 1) / 2) * 255
    for i in range(8):
        rand_idx = np.random.randint(0,48)
        # Convert the numpy array to an image
        pil_image = Image.fromarray(X[rand_idx].astype('uint8'))
        # Save the image to the static folder
        image_path = os.path.join("static", f"image_{i+1}.png")
        pil_image.save(image_path)
        imgs_path.append(image_path)


    # X = create_plot(X,1)
    # picking a random img index to output
    # Y = X[rand_idx]
    # print(X[rand_idx].shape)
    # # Upscale the image by a factor of 2 using nearest neighbor interpolation
    # Y  = cv2.resize(Y, (256, 256), interpolation=cv2.INTER_NEAREST)
    # pyplot.imsave('single_predicted_image.png', Y, dpi=70)
    # return send_file('single_predicted_image.png', mimetype='image/png')
    return render_template('predict_mul.html', imgs_path=imgs_path)


if __name__ == "__main__":
    app.debug = True
    app.run(port=9899,debug=True)