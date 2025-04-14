import os
import torch
import pandas as pd
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms

from config import *
from models.classifier import ImprovedClassifier
from utils.feature_extraction import extract_clip_features_from_image
from utils.matching import find_most_similar_within_indices, generate_geolocation_csv
from utils.datasets import ImageDataset  

def process_images(image_dir_path, output_folder, device, batch_size=16):
    # Load model
    classifier = ImprovedClassifier(INPUT_DIM, NUM_CLASSES).to(device)
    classifier.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    classifier.eval()

    # Load features and metadata
    precomputed_features = np.load(PRECOMPUTED_FEATURES_PATH)
    precomputed_df = pd.read_csv(PRECOMPUTED_METADATA_PATH)

    # Setup dataset and dataloader
    transform = transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.ToTensor()
])
    dataset = ImageDataset(image_dir_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    results = []
    for idx, (images, image_names) in enumerate(dataloader):
        print(f'Processing batch {idx+1}/{len(dataloader)}')
        images = images.to(device)
        # Extract CLIP features for batch
        features_batch = extract_clip_features_from_image(images)  # Should return a tensor [B, D]

        # Predict with classifier
        outputs = classifier(features_batch)
        predictions = torch.nn.functional.softmax(outputs, dim=1)
        top3_indices_batch = torch.topk(predictions, k=3, dim=1).indices.tolist()

        features_np = features_batch.cpu().numpy()

        for i in range(len(image_names)):
            top3 = top3_indices_batch[i]
            best_img_id, similarity_score = find_most_similar_within_indices(
                features_np[i], top3, precomputed_df, precomputed_features
            )

            results.append({
                'image': image_names[i],
                'matched_image': best_img_id,
                'similarity': similarity_score
            })

    # Save results
    os.makedirs(output_folder, exist_ok=True)
    result_file_path = os.path.join(output_folder, f'predictions_{os.path.basename(image_dir_path)}.csv')
    pd.DataFrame(results).to_csv(result_file_path, index=False)

    geolocation_file_path = os.path.join(output_folder, f'geolocation_results_{os.path.basename(image_dir_path)}.csv')
    generate_geolocation_csv(results, PRECOMPUTED_METADATA_PATH, 'ressources/image_coordinates_im2_gps2k.csv', geolocation_file_path)

    print(f"Results saved in {output_folder}")

def parse_args():
    parser = argparse.ArgumentParser(description="Image processing with classification and geolocation matching")
    parser.add_argument('image_dir', type=str, help="Directory path for input images")
    parser.add_argument('output_dir', type=str, help="Directory path for saving output results")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help="Device for processing (default: fetch from config)")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for image processing")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = args.device if args.device else DEVICE
    process_images(args.image_dir, args.output_dir, device, batch_size=args.batch_size)
