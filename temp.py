import pandas as pd
from utils.matching import generate_geolocation_csv
from config import *
import os

image_dir_path = "datasets/im2gps_test2k"

results = pd.read_csv('results/predictions_im2gps_test2k.csv')
generate_geolocation_csv(results, PRECOMPUTED_METADATA_PATH, 'ressources/image_coordinates_im2_gps2k.csv', f'results/geolocation_results_{os.path.basename(image_dir_path)}.csv')