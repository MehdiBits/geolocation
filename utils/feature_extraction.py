import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from config import DEVICE
from numpy import random

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


def extract_clip_features_from_image(image):
    """
    Extract feature vector from an image using CLIP.

    Args:
        image (PIL Image): Image or (N x N x 3) array.
    
    Returns:
        Tensor: Extracted feature vector.
    """
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    
    return features


def rotate_image_randomly_symmetry(image, max_angle=20):

    # Extend the image by mirroring
    width, height = image.size
    extended_image = Image.new("RGB", (width * 3, height * 3))

    # Paste the mirrored sections
    extended_image.paste(image, (width, height))  # Center
    extended_image.paste(image.transpose(Image.FLIP_LEFT_RIGHT), (0, height))  # Left
    extended_image.paste(image.transpose(Image.FLIP_LEFT_RIGHT), (width * 2, height))  # Right
    extended_image.paste(image.transpose(Image.FLIP_TOP_BOTTOM), (width, 0))  # Top
    extended_image.paste(image.transpose(Image.FLIP_TOP_BOTTOM), (width, height * 2))  # Bottom
    extended_image.paste(image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM), (0, 0))  # Top-left
    extended_image.paste(image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM), (width * 2, 0))  # Top-right
    extended_image.paste(image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM), (0, height * 2))  # Bottom-left
    extended_image.paste(image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM), (width * 2, height * 2))  # Bottom-right

    # Generate a random angle between -max_angle and max_angle
    angle = random.randint(-max_angle, max_angle)

    # Rotate the extended image
    rotated_image = extended_image.rotate(angle, resample=Image.BICUBIC, center=(width * 3 // 2, height * 3 // 2))

    # Crop back to the original size
    rotated_image = rotated_image.crop((width, height, width * 2, height * 2))

    return rotated_image, angle