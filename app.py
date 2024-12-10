from flask import Flask, request, render_template
import os
import numpy as np
import cv2
import base64
from ultralytics import YOLO
from PIL import Image

# Directory where the models are saved
save_dir = 'Prepare models\\trained models'

# Load the models
loaded_models = {}
for model_name in ['yolov8', 'yolov9', 'yolov10', 'yolov11']:
    model_path = os.path.join(save_dir, f'{model_name}_model.pt')
    model = YOLO(model_path)
    loaded_models[model_name] = model

# Create flask application
app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/')

# Define route for home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the uploaded image file and the selected model name from the form data
        image_file = request.files['image']
        model_name = request.form['model']

        # Choose model based on the user input name
        model = loaded_models[model_name]

        # Open the image and perform object detection
        image = Image.open(image_file.stream)
        results = model(image, conf=0.8)

        # Filter results to keep only the box with the highest confidence
        if results[0].boxes:
            boxes = results[0].boxes
            max_conf_box = max(boxes, key=lambda box: box.conf)
            results[0].boxes = [max_conf_box]

        # Draw the detected box on the image
        detected_image = results[0].plot(line_width=3, font_size=16)
        # Convert the detected image to a numpy array
        detected_image_np = np.array(detected_image)

        # Encode the image to base64 to be shown in index.html file
        buffer = cv2.imencode('.png', detected_image_np)[1]
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Render the detected image in the template
        return render_template('index.html', detected_image=image_base64)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
