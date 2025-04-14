import torch

BEST_PARAMS = {
    'lr': 3.8745e-05,
    'weight_decay': 2.9289e-06,
    'dropout_rate': 0.225,
    'batch_size': 128,
}

NUM_CLASSES = 35664  # Number of geographical cells
INPUT_DIM = 768      # Feature vector dimension from CLIP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "geolocation/src/ressources/no_std_14-10(lvl8).pth"  # Path to classifier weights
PRECOMPUTED_FEATURES_PATH = "geolocation/src/ressources/all_features_from_batches(concat).npy"
PRECOMPUTED_METADATA_PATH = "geolocation/src/ressources/index_14-10.csv"