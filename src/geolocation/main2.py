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
    # Load model
    model = torch.load(model_path, map_location=device) 
    num_classes = model['fc4.bias'].shape[0]
    classifier = ImprovedClassifier(INPUT_DIM, num_classes).to(device)
    classifier.load_state_dict(model)
    classifier.eval()
    print(f"Model loaded from {model_path}")
    print('----------------------')
    # Load features and metadata
    with importlib.resources.path("geolocation.ressources", PRECOMPUTED_FEATURES_PATH) as path:
        precomputed_features = np.load(path)
    with importlib.resources.path("geolocation.ressources", PRECOMPUTED_METADATA_PATH) as path:
        precomputed_df = pd.read_csv(path)
    

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
    generate_geolocation_csv(pd.DataFrame(results), PRECOMPUTED_METADATA_PATH, coordinates_file, geolocation_file_path)

    print(f"Results saved in {output_folder}")

def parse_args():
    parser = argparse.ArgumentParser(description="Image processing with classification and geolocation matching")
    parser.add_argument('image_dir', type=str, help="Directory path for input images")
    parser.add_argument('output_dir', type=str, help="Directory path for saving output results")
    parser.add_argument('coordinates_file', type=str, help="Path to the true coordinates file")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], help="Device for processing (default: fetch from config)")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for image processing")
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help="Path to the model weights")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = args.device if args.device else DEVICE
    process_images(args.image_dir, args.output_dir, args.coordinates_file, device, batch_size=args.batch_size, model_path=args.model_path)
