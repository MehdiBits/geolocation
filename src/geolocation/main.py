import os
import torch
import pandas as pd
import numpy as np
import argparse
from config import *
from models.classifier import ImprovedClassifier
from utils.feature_extraction import extract_clip_features
from utils.matching import find_most_similar_within_indices, generate_geolocation_csv


def process_images(image_dir_path, output_folder, device):
    # Loading model
    classifier = ImprovedClassifier(INPUT_DIM, NUM_CLASSES).to(device)
    classifier.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    classifier.eval()

    # Load database precomputed CLIP features and metadata
    precomputed_features = np.load(PRECOMPUTED_FEATURES_PATH)
    precomputed_df = pd.read_csv(PRECOMPUTED_METADATA_PATH)

    
    results = []

    img_dir = [img for img in os.listdir(image_dir_path) 
                if img.lower().endswith((".jpg", ".png", ".jpeg"))]  # Only select images

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
    os.makedirs(output_folder, exist_ok=True)
    result_file_path = os.path.join(output_folder, f'predictions_{os.path.basename(image_dir_path)}.csv')
    pd.DataFrame(results).to_csv(result_file_path, index=False)

    geolocation_file_path = os.path.join(output_folder, f'geolocation_results_{os.path.basename(image_dir_path)}.csv')
    generate_geolocation_csv(results, PRECOMPUTED_METADATA_PATH, geolocation_file_path, ground_truth_file='ressources/image_coordinates_im2_gps2k.csv')

    print(f"Results saved in {output_folder}")

# Define the argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Image processing with classification and geolocation matching")
    parser.add_argument('image_dir', type=str, help="Directory path for input images")
    parser.add_argument('output_dir', type=str, help="Directory path for saving output results")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help="Device for processing (default: fetch from config)")

    return parser.parse_args()

# Main entry point of the script
if __name__ == "__main__":
    args = parse_args()
    # If device is not specified, use the one from config
    device = args.device if args.device else DEVICE

    process_images(args.image_dir, args.output_dir, device)
