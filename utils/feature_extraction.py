import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from config import DEVICE

# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def extract_clip_features(image_path):
    """
    Extract feature vector from an image using CLIP.

    Args:
        image_path (str): Path to the image.
    
    Returns:
        Tensor: Extracted feature vector.
    """
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    
    return features
