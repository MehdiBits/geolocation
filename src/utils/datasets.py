from torch.utils.data import Dataset
from PIL import Image
import os

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