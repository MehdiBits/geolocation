import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.path import Path
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

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



def find_most_similar_in_contour(target_feature, precomputed_df, precomputed_features, contour_paths):
    """
    Finds the most visually similar image within the contour-defined region.

    Args:
        target_feature (np.array): Feature vector of the input image.
        precomputed_df (pd.DataFrame): Metadata of precomputed images (must contain 'latitude' and 'longitude').
        precomputed_features (np.array): Feature vectors of precomputed images.
        contour_paths (list): List of Matplotlib Path objects defining the region.

    Returns:
        (str, float): Most similar image ID and similarity score.
    """
    if not contour_paths:
        return "No match", 0

    # Convert LAT/LON to NumPy array for vectorized processing
    points = np.column_stack((precomputed_df['LAT'].values, precomputed_df['LON'].values))  # (N, 2)

    # Vectorized check: Find points inside the contour
    inside_mask = contour_paths.contains_points(points)

    if not np.any(inside_mask):  # No images inside the contour
        return "No match", 0

    # Filter data based on the mask
    filtered_features = precomputed_features[inside_mask]
    filtered_img_ids = precomputed_df['IMG_ID'].values[inside_mask]

    # Compute cosine similarity in a vectorized way
    similarities = cosine_similarity(target_feature.reshape(1, -1), filtered_features)[0]
    most_similar_index = np.argmax(similarities)

    return filtered_img_ids[most_similar_index], similarities[most_similar_index]

def interest_zones(gps_cord, gmm_n=3, percentile=95, grid_npoints=200):
    """
    Identifies areas of interest based on GPS coordinates using Gaussian Mixture Models (GMM).

    Parameters:
        gps_cord (array-like): A list or array of GPS coordinates, where each entry is a tuple or list 
                            containing latitude and longitude values.
        gmm_n (int, optional): The number of components for the Gaussian Mixture Model. Default is 3.
        percentile (float): The percentile value used to determine the contour level for the areas of interest.
        grid_npoints (int, optional): The number of points to use for the grid in each dimension. Default is 200.

    Returns:
        matplotlib.path.Path: A path object representing the contour of the area of interest.
    """
    df = pd.DataFrame(gps_cord, columns=["latitude", "longitude"])

    min_lat, max_lat = np.percentile(df['latitude'], 1), np.percentile(df['latitude'], 99)
    min_lon, max_lon = np.percentile(df['longitude'], 1), np.percentile(df['longitude'], 99)

    X = df[["longitude", "latitude"]].values
    gmm = GaussianMixture(n_components=gmm_n, covariance_type='full', random_state=42)
    gmm.fit(X)

    latitudes = np.linspace(min_lat, max_lat, grid_npoints)
    longitudes = np.linspace(min_lon, max_lon, grid_npoints)
    grid_lat, grid_lon = np.meshgrid(latitudes, longitudes)
    grid_points = np.vstack([grid_lon.ravel(), grid_lat.ravel()]).T

    grid_probs = np.exp(gmm.score_samples(grid_points))

    contour = plt.contour(grid_lat, grid_lon, grid_probs.reshape(grid_npoints, grid_npoints), levels=[np.percentile(grid_probs, percentile)])

    paths = contour.get_paths()[0]
    
    return paths

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