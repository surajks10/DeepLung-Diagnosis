from flask import Flask, render_template, request
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

model = load_model('Masked_CNN_pneumonia.hdf5')
model1 = load_model('malaria_masked_cnn.h5')

ALLOWED_EXTENSIONS_PNEUMONIA = {'jpg', 'jpeg'}
ALLOWED_EXTENSIONS_MALARIA = {'png'}
MALARIA_IMAGE_SIZE = (800, 600)

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if file.filename == '':
        return "Please upload a file"
    if not allowed_file(file.filename, ALLOWED_EXTENSIONS_PNEUMONIA):
        return render_template('index.html', pneumonia_error="Please Provide a Valid Image")
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    print(prediction)
    predicted_class = np.argmax(prediction)
    if predicted_class == 0:
        result = f'Normal as confidence is {prediction[0][0]}'
    else:
        result = f'Pneumonia Detected as confidence is {prediction[0][1]}'

    return render_template('result.html', prediction_results=result)

@app.route('/malaria_predict', methods=['POST'])
def malaria_predict():
    if request.method == "POST":
        image_file = request.files['image']
        
        # Check if the file is a PNG image
        if not allowed_file(image_file.filename, ALLOWED_EXTENSIONS_MALARIA):
            return render_template('index.html', malaria_error="Please upload a PNG image.")
        
        # Read the image
        image = Image.open(io.BytesIO(image_file.read()))
        
        # Check the dimensions of the image
        if image.size != MALARIA_IMAGE_SIZE:
            return render_template('index.html', malaria_error="Please upload an valid image.")
        
        # Convert image to numpy array and preprocess
        image = np.array(image)
        image = cv2.resize(image, (50, 50)) / 255.0
        mask = np.ones_like(image)
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        # Make prediction
        prediction = model1.predict([image, mask])
        
        # Create the result
        if prediction[0][0] > 0.5:
            result = f'Malaria Detected as confidence is {prediction[0][0] * 100:.2f}%'
        else:
            result = f'Normal as confidence is {prediction[0][1] * 100:.2f}%'
        
        return render_template('result.html', prediction_result=result)

if __name__ == '__main__':
    app.run(debug=True)
