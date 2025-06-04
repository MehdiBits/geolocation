# Geolocation from a single image


This research project focuses on geolocating images. The goal is, from a single input image, output likely GPS coordinates of where the image was taken.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)

## Installation

You can install this repo using one the following:

### (Recommended) Using uv

```bash
git clone https://github.com/MehdiBits/geolocation.git
cd geolocation
uv pip install -e .
```

### Wihtout uv
```bash
git clone https://github.com/MehdiBits/geolocation.git
cd geolocation
pip install -e .
```

## Usage
To directly use the model, provided you have a checkpoint, the precomputed features and the metadata, you can directly use the following:

### (Recommended) Using uv
```bash
uv run src/geolocation/main2.py path/to/image_folder path/to/output.csv
```

### Wihtout uv
```
python -m geolocation/src/main2.py --input path/to/image_folder --output path/to/output.csv
```

This will output a csv file containing three columns, one being the image name and the two others the predicted latitude and longitude of the image.

A few optional parameters exists:

* --device: Specify which device is used, currently supported devices are "cpu", "cuda" or "mps".
* --batch_size: Size of the batch processed at the same time by the algorithm.
* --model_path: Path to the model used by the neural network classifier.
* --coordinates_file: Csv file containing the real coordinates of the images, mainly used to benchmark the method.


A trained checkpoint along with precomputed features and the metadata are available to [download](https://zenodo.org/records/15593465), they need to then be refered to in the config.py file or, in the case of the model path, be passed with the --model_path argument.

## Features
- Uses a neural network for rough prediction of geolocation which is then refined using a database of images giving (lat, lon) prediction.
-  The method is lightweight, capable of predicting high numbers of images in a short amount of time.
- Accuracy varies on the dataset but a high accuracy is reached for localizable images.

