# Face Detection using YOLO Models

The Face Detection project leverages the YOLO (You Only Look Once) family of models (YOLOv8, YOLOv9, YOLOv10, YOLOv11) to detect faces in images. 
This project aims to provide an efficient and accurate solution for face detection, which is crucial for various applications such as security, surveillance, and human-computer interaction.

![Image about the final project](<Human Faces Detection.png>)

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Features
- **Face Detection**: Uses YOLO with custom-trained models to detect faces in images.
- **Interactive Web Application**: Built using Flask to upload images and display predictions.
- **Bounding Box Visualization**: Shows predicted bounding boxes around detected faces with confidence scores.

## Requirements
- Python 3.7+
- Flask
- Ultralytics YOLO
- OpenCV
- Pandas
- Numpy
- Matplotlib
- Pillow
- scikit-learn
- PyYAML

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/eslammohamedtolba/Human-Faces-Detection-using-YOLO-V8-V9-V10-V11.git
   cd Human-Faces-Detection-using-YOLO-V8-V9-V10-V11
   ```

2. **Install the required dependencies**:
   ```bash
   pip install Flask ultralytics opencv-python pandas numpy matplotlib scikit-learn pyyaml
   ```

## Usage

1. **Run the Flask app**:
   ```bash
   python app.py
   ```

2. **Access the web application** at `http://127.0.0.1:5000/`.

3. **Upload an image** and **choose the YOLO model (YOLOv8, YOLOv9, YOLOv10, YOLOv11)** to detect faces. The image will be processed, and bounding boxes will be drawn around detected faces.

## Project Structure

- `app.py`: Main Flask application file.
- `Prepare model/Face Detection using YOLO.ipynb`: Jupyter notebook for training and evaluation.
- `Prepare model/trained_models/`: Directory containing saved state dictionaries of the trained models.
- `templates/index.html`: HTML template for the web interface.
- `static/style.css`: CSS file for styling the web interface.

## Contributing

Contributions are welcome! Here’s how you can contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.
