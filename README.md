# Geolocation from a single image


This research project focuses on geolocating images. The goal is, from a single input image, output likely GPS coordinates of where the image was taken.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)

## Installation

You can install this repo using the following:

```bash
git clone https://github.com/MehdiBits/geolocation.git
cd geolocation
pip install -e .
```

## Usage
To directly use the model, provided you have a checkpoint, you can directly use the following:

```
python -m geolocation/src/main.py --input path/to/image_folder --output path/to/output.csv
```

This will output a csv file containing two columns, one being the image name and the other the predicted angle.

A trained checkpoint is available to [download](https://drive.google.com/file/d/1myCTKFY4jt1xVDdf0EM2H3Vt0TCKVS5W/view?usp=drive_link), it needs to then be refered to in the config.py file.

## Features
- Rotation Angle Prediction: Uses a Convolutional Neural Network based on EfficientNet-B3 to predict the rotation angle of an image.
- Regression Output: The CNN outputs two numbers representing the sine and cosine of the angle, encapsulating the periodic nature of angles.
- The default model was specifically trained to handle rotation angle ranging from $[-30°, 30°]$.

