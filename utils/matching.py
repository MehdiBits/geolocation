import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def haversine(lat1, lon1, lat2, lon2):
    """
    Compute the great-circle distance (in km) between two points on Earth.

    Args:
        lat1, lon1: Latitude and Longitude of first point.
        lat2, lon2: Latitude and Longitude of second point.

    Returns:
        float: Distance in kilometers.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return 6371 * c  # Earth's radius in km

def find_most_similar_within_indices(target_feature, target_indices, precomputed_df, precomputed_features):
    """
    Finds the most visually similar image within given cell indices.

    Args:
        target_feature (np.array): Feature vector of the input image.
        target_indices (list): Top-3 predicted cell indices.
        precomputed_df (pd.DataFrame): Metadata of precomputed images.
        precomputed_features (np.array): Feature vectors of precomputed images.

    Returns:
        (str, float): Most similar image ID and similarity score.
    """
    best_similarity = -1
    best_img_id = None

    for target_index in target_indices:
        # Filter dataset to images in the predicted cells
        filter_mask = (precomputed_df['CellId_int'] == target_index)
        filtered_features = precomputed_features[filter_mask]
        filtered_img_ids = precomputed_df[filter_mask]['IMG_ID'].tolist()

        if filtered_features.size == 0:
            continue  # Skip if no images in this cell

        # Compute cosine similarity
        
        similarities = cosine_similarity(target_feature, filtered_features)[0]
        most_similar_index = np.argmax(similarities)

        if similarities[most_similar_index] > best_similarity:
            best_similarity = similarities[most_similar_index]
            best_img_id = filtered_img_ids[most_similar_index]
    return (best_img_id, best_similarity) if best_img_id else ("No match", 0)

def generate_geolocation_csv(results_df, index_file, ground_truth_file, output_file):
    """
    Generate a CSV with estimated and actual geolocation data along with distance.
    
    Args:
        results_df (pd.DataFrame): Dataframe containing input images and their best match.
        index_file (str): Path to CSV file with geolocation data for matched images.
        ground_truth_file (str): Path to CSV file with true locations of input images.
        output_file (str): Path to save the final CSV.
    """
    # Load database geolocation data
    db_locations = pd.read_csv(index_file)[['IMG_ID', 'LAT', 'LON']]
    
    # Merge to get estimated geolocation (from best match in database)
    merged_df = results_df.merge(db_locations, left_on='matched_image', right_on='IMG_ID', how='left')
    merged_df.rename(columns={'LAT': 'est_LAT', 'LON': 'est_LON'}, inplace=True)
    merged_df.drop(columns=['IMG_ID'], inplace=True)
    
    # Load ground truth locations for input images
    gt_locations = pd.read_csv(ground_truth_file)[['IMG_ID', 'LAT', 'LON']]
    
    # Merge to get actual geolocation of input images
    merged_df = merged_df.merge(gt_locations, left_on='image', right_on='IMG_ID', how='left')
    merged_df.rename(columns={'LAT': 'true_LAT', 'LON': 'true_LON'}, inplace=True)
    merged_df.drop(columns=['IMG_ID'], inplace=True)
    
    # Compute distance
    merged_df['distance_km'] = merged_df.apply(
        lambda row: haversine(row['true_LAT'], row['true_LON'], row['est_LAT'], row['est_LON']), axis=1
    )
    
    # Save to CSV
    merged_df[['image', 'est_LAT', 'est_LON', 'true_LAT', 'true_LON', 'distance_km']].to_csv(output_file, index=False)
    print(f"CSV file saved: {output_file}")
    
    return merged_df