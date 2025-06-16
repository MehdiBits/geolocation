import os
import torch
import pandas as pd
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
import importlib.resources

from geolocation.config import *
from geolocation.models.classifier import ImprovedClassifier
from geolocation.utils.feature_extraction import extract_clip_features_from_image
from geolocation.utils.matching import find_most_similar_within_indices, generate_geolocation_csv
from geolocation.utils.datasets import EmbeddingDataset

def process_images(image_dir_path, output_folder, coordinates_file, device, batch_size=16, model_path=MODEL_PATH):
    """
    Process images in the specified directory, classify them using a pre-trained model,
    and find the most similar images in a precomputed database. Generate a CSV file 
    with the geolocation results for each image. If a coordinates file is provided,
    it will also include the true coordinates for comparison.

    Args:
        image_dir_path (str): Path to the directory containing images to process.
        output_folder (str): Directory where results will be saved.
        coordinates_file (str): Path to the file containing true coordinates for geolocation. Optional, default is None.
        device (str): Device to run the model on ('cpu', 'cuda', or 'mps'). O
        batch_size (int): Number of images to process in each batch.
        model_path (str): Path to the pre-trained model weights.
    """
        
    # Load model
    try:
        model = torch.load(model_path, map_location=device) 
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {model_path}, please specify a valid path in either the CONFIG.PY file or pass it as an argument.")
    num_classes = model['fc4.bias'].shape[0]
    classifier = ImprovedClassifier(INPUT_DIM, num_classes).to(device)
    classifier.load_state_dict(model)
    classifier.eval()
    print(f"Model loaded from {model_path}")
    print('----------------------')
    # Load features and metadata
    with importlib.resources.path("geolocation.ressources", PRECOMPUTED_FEATURES_PATH) as path:
        try:
            precomputed_features = np.load(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Precomputed features file not found at {path}, please specify a valid path in the CONFIG.PY file.")
    with importlib.resources.path("geolocation.ressources", PRECOMPUTED_METADATA_PATH) as path:
        try:
            precomputed_df = pd.read_csv(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Precomputed metadata file not found at {path}, please specify a valid path in the CONFIG.PY file.")


    # Setup dataset and dataloader
    transform = transforms.Compose([
    transforms.ToTensor()
])
    dataset = EmbeddingDataset(image_dir_path, transform=transform, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    results = []
    for idx, (embeddings, image_names) in enumerate(dataloader):
        print(f'Processing batch {idx+1}/{len(dataloader)}')
        
        # Predict with classifier
        outputs = classifier(embeddings)
        predictions = torch.nn.functional.softmax(outputs, dim=1)
        top3_indices_batch = torch.topk(predictions, k=3, dim=1).indices.tolist()

        features_np = embeddings.cpu().numpy()

        for i in range(len(image_names)):
            top3 = top3_indices_batch[i]
            # Find most similar image in the database within the top-3 predicted cell indices
            best_img_id, similarity_score = find_most_similar_within_indices(
                features_np[np.newaxis, i, :], top3, precomputed_df, precomputed_features
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
    generate_geolocation_csv(pd.DataFrame(results), PRECOMPUTED_METADATA_PATH, geolocation_file_path, ground_truth_file=coordinates_file)

    print(f"Results saved in {output_folder}")

def parse_args():
    parser = argparse.ArgumentParser(description="Image processing with classification and geolocation matching")
    parser.add_argument('image_dir', type=str, help="Directory path for input images")
    parser.add_argument('output_dir', type=str, help="Directory path for saving output results")
    parser.add_argument('--coordinates_file', type=str, default=None, help="Path to the true coordinates file")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], help="Device for processing (default: fetch from config)")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for image processing")
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help="Path to the model weights")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = args.device if args.device else DEVICE
    process_images(args.image_dir, args.output_dir, args.coordinates_file, device, batch_size=args.batch_size, model_path=args.model_path)
