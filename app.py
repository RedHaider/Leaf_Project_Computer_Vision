from flask import Flask, render_template, Response, url_for,  redirect, request,  send_from_directory
import os
import cv2
from retinaface import RetinaFace
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

yolo_v_model = YOLO('lead_yolo_V8.pt')

# Initialize video capture

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

#custom cnn double detection
@app.route('/custom_model')
def custom_model():
    return render_template('custom_model.html')

@app.route('/to_custom_model')
def redirect_to_custom_model():
    return redirect(url_for('custom_model'))
###################################################

#

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(port=8080, debug=True)
