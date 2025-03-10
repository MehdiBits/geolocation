import os
import torch
import pandas as pd
import numpy as np
from config import *
from models.classifier import ImprovedClassifier
from utils.feature_extraction import extract_clip_features
from utils.matching import find_most_similar_within_indices, generate_geolocation_csv


image_dir_path = "datasets/im2gps_test2k"

# Load classifier
classifier = ImprovedClassifier(INPUT_DIM, NUM_CLASSES).to(DEVICE)
classifier.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
classifier.eval()

# Load precomputed features
precomputed_features = np.load(PRECOMPUTED_FEATURES_PATH)
precomputed_df = pd.read_csv(PRECOMPUTED_METADATA_PATH)

# Process images
results = []

img_dir = [img for img in os.listdir(image_dir_path) 
            if img.lower().endswith((".jpg", ".png", ".jpeg"))] # Only select images

image_dir_size = len(img_dir)
for idx, img_name in enumerate(img_dir):
    print(f'Processing image n°{idx}/{image_dir_size}')
    image_path = os.path.join(image_dir_path, img_name)
    features = extract_clip_features(image_path) 

    # Predict cell indices
    output = classifier(features)
    prediction = torch.nn.functional.softmax(output, dim=1)
    top3_indices = torch.topk(prediction, k=3, dim=1).indices.squeeze(0).tolist()

    # Find best match
    best_img_id, similarity_score = find_most_similar_within_indices(features.cpu().numpy(), top3_indices, precomputed_df, precomputed_features)
    
    results.append({'image': img_name, 'matched_image': best_img_id, 'similarity': similarity_score})

# Save results
pd.DataFrame(results).to_csv(f'results/predictions_{os.path.basename(image_dir_path)}.csv', index=False)

generate_geolocation_csv(results, PRECOMPUTED_METADATA_PATH, 'ressources/image_coordinates_im2_gps2k.csv', f'results/geolocation_results_{os.path.basename(image_dir_path)}.csv')


