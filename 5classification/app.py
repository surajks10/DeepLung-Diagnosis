import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)

# Load the trained model
MODEL_NAME = 'Lungdisease_model.h5'
model = load_model("C:/Users/91807/dummy/abc/Lungdisease-0.001-2conv-basic/model.h5")

# Define image size
IMG_SIZE = 50

# Define labels
labels = ['Cancer', 'Viral Pneumonia', 'Covid', 'Tuberculosis', 'Normal']

# Function to preprocess image
def preprocess_image(image):
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, axis=0)
    return img

# Function to check if the image is an X-ray image
def is_xray_image(image):
    height, width, _ = image.shape
    expected_height = 224
    expected_width = 224
    return height == expected_height and width == expected_width


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = cv2.imdecode(np.fromstring(request.files['file'].read(), np.uint8), cv2.IMREAD_COLOR)
            if not is_xray_image(img):
                return "Invalid image. Please upload an valid X-ray image."
            img = preprocess_image(img)
            prediction = model.predict(img)
            result = labels[np.argmax(prediction)]
            return result

if __name__ == '__main__':
    app.run(debug=True)
