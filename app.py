from flask import Flask, render_template, Response, url_for,  redirect, request,  send_from_directory
import os
import cv2
from retinaface import RetinaFace
from ultralytics import YOLO
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
import keras
import tensorflow as tf
print(keras.__version__)
print('##################TensorFlow version: ' + tf.__version__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB 
disease_model = tf.keras.models.load_model('trained_plant_disease_model.h5')
leaf_model = tf.keras.models.load_model('trained_plant_leaf_model.h5')


yolo_v_model = YOLO('lead_yolo_V8.pt')

# Classes for predictions
disease_classes = [
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)__Common_rust', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Jackfruit___Algal_Leaf_Spot', 'Jackfruit___Black_Spot', 'Jackfruit___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Rice_Bacterialblight', 'Rice_Brownspot', 'Rice_Leafsmut', 'Rose_Healthy_Leaf',
    'Rose_Rust', 'Rose_sawfly_Rose_slug', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

leaf_classes = [
    'Corn_(maize)___Leaf', 'Jackfruit___Leaf', 'Potato___Leaf',
    'Rice___Leaf', 'Rose___Leaf', 'Tomato___Leaf'
]

# Function to preprocess an image
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(128, 128))  # Resize to model's expected input size
    image = img_to_array(image)  # Convert to array
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

#custom cnn double detection
###################################################
@app.route('/custom_model')
def custom_model():
    return render_template('custom_model.html')

@app.route('/to_custom_model')
def redirect_to_custom_model():
    return redirect(url_for('custom_model'))
###################################################
#################################################
@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    predictions = None
    if request.method == 'POST' and 'image' in request.files:
        file = request.files['image']
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            image = preprocess_image(filename)
            # Predict using the loaded .h5 models
            disease_prediction_index = np.argmax(disease_model.predict(image))
            leaf_prediction_index = np.argmax(leaf_model.predict(image))
            predictions = [
                {'label': disease_classes[disease_prediction_index], 'type': 'Disease'},
                {'label': leaf_classes[leaf_prediction_index], 'type': 'Leaf Type'}
            ]
            image_path = os.path.join('uploads', file.filename)
            return render_template('upload_and_results.html', predictions=predictions, image_path=image_path)

    return render_template('upload_and_results.html', predictions=predictions, image_path=None)

# Other routes remain unchanged


#################################################
@app.route('/detect_leaf_yolo', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return "No file part"
    file = request.files['image']
    if file.filename == '':
        return "No Selected File"
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Load and process the image
        image = cv2.imread(filepath)

        # Run YOLO model inference
        results = yolo_v_model(image)

        # Extract predictions
        predictions = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy.cpu().numpy()[0]
                conf = box.conf.cpu().numpy()[0]
                cls = box.cls.cpu().numpy()[0]
                label = yolo_v_model.names[int(cls)]
                predictions.append({
                    'label': label,
                    'confidence': round(float(conf), 2),
                    'coordinates': xyxy.tolist()
                })

        # Render predictions on the custom_model.html page
        return render_template(
            'custom_model.html',
            predictions=predictions,
            image_path=url_for('uploaded_file', filename=file.filename)
        )



@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



    
#################################################
#Login
@app.route('/')
def index():
    return render_template('login.html')

@app.route('/redirect_to_login')
def redirect_to_login():
    return redirect(url_for('index'))


###################################################

#Dashboard
@app.route('/dashboard')
def dashboard():
    return render_template('index.html')

@app.route('/redirect_to_dashboard')
def redirect_to_dashboard():
    return redirect(url_for('dashboard'))

###################################################

#yolo v detection of leaf
@app.route('/yolov')
def yolov():
    return render_template('yolov.html')

@app.route('/route_to_yolov')
def redirect_to_yolov():
    return redirect(url_for('yolov'))

###################################################


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(port=8080, debug=True)
