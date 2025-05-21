from torch.utils.data import Dataset
from PIL import Image
import os

from geolocation.utils.feature_extraction import extract_clip_features_from_image
from geolocation.config import DEVICE

class ImageDataset(Dataset):
    """
    A standard PyTorch Dataset class for loading images.

    Args:
        image_dir (str): Path to the directory containing image files.
        transform (callable, optional): A function/transform to apply to the images.

    Attributes:
        image_paths (list): List of file paths to the images in the specified directory.
        transform (callable): Transform function to apply to the images.
    """
    def __init__(self, image_dir, transform=None):
        self.image_paths = [
            os.path.join(image_dir, img) 
            for img in os.listdir(image_dir) 
            if img.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)


        image_name = os.path.basename(img_path)
        return image, image_name
    
class EmbeddingDataset(Dataset):
    """
    A Pytorch Dataset for image Embedding obtained through CLIP.

    Args:
        image_dir (str): Path to the directory containing image files.
        transform (callable, optional): A function/transform to apply to the images.
        device (str): Device to use for processing ("cuda", "cpu" or "mps"). Defaults to DEVICE from config.

    Attributes:
        image_paths (list): List of file paths to the images in the specified directory.
        transform (callable): Transform function to apply to the images.
    """
    def __init__(self, image_dir, transform=None, device=DEVICE):
        self.image_paths = [
            os.path.join(image_dir, img) 
            for img in os.listdir(image_dir) 
            if img.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        

        if self.transform:
            image = self.transform(image)
        image.to(self.device)
        embedding = extract_clip_features_from_image(image, device=self.device).to(self.device).squeeze()
        image_name = os.path.basename(img_path)
        return embedding, image_name