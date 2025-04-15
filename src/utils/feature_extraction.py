import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from numpy import random

from config import DEVICE

class CLIPModelSingleton:
    """
    Singleton class to load and store a single instance of CLIPModel. This prevents loading multiple time the model.
    """
    _instance = None

    def __new__(cls, device=DEVICE):
        if cls._instance is None:
            print(f"Loading CLIP model on {device}...")
            cls._instance = super(CLIPModelSingleton, cls).__new__(cls)
            cls._instance.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
            cls._instance.device = device
        return cls._instance

class CLIPProcessorSingleton:
    """
    Singleton class to load and store a single instance of CLIPProcessor. This prevents loading multiple time the processor.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print("Loading CLIP processor...")
            cls._instance = super(CLIPProcessorSingleton, cls).__new__(cls)
            cls._instance.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", do_rescale=False)
        return cls._instance

def extract_clip_features(image_path, device=DEVICE):
    """
    Extracts a feature vector from an image using CLIP.

    Args:
        image_path (str): Path to the image.
        device (str, optional): Device to use ("cuda" or "cpu"). Defaults to DEVICE from config.

    Returns:
        torch.Tensor: Extracted feature vector.
    """
    instance_model = CLIPModelSingleton(device)
    instance_processor = CLIPProcessorSingleton()

    image = Image.open(image_path)
    inputs = instance_processor.processor(images=image, return_tensors="pt").to(instance_model.device)
    
    with torch.no_grad():
        features = instance_model.model.get_image_features(**inputs)
    
    return features

def extract_clip_features_from_image(image, device=DEVICE):
    """
    Extracts a feature vector from a PIL image using CLIP.

    Args:
        image (PIL.Image): Image to extract features from.
        device (str, optional): Device to use ("cuda" or "cpu"). Defaults to DEVICE from config.

    Returns:
        torch.Tensor: Extracted feature vector.
    """
    instance_model = CLIPModelSingleton(device)
    instance_processor = CLIPProcessorSingleton()
    
    inputs = instance_processor.processor(images=image, return_tensors="pt").to(instance_model.device)
    
    with torch.no_grad():
        features = instance_model.model.get_image_features(**inputs)
    
    return features

def rotate_image_symmetry(image, angle):
    """
    Rotates an image randomly while maintaining symmetry by extending it through mirroring.

    Args:
        image (PIL.Image): Input image.
        angle (float): Rotation angle in degrees.

    Returns:
        tuple: (Rotated PIL.Image, float angle used for rotation)
    """
    width, height = image.size

    extended_image = Image.new("RGB", (width * 3, height * 3))
    extended_image.paste(image, (width, height))
    extended_image.paste(image.transpose(Image.FLIP_LEFT_RIGHT), (0, height))
    extended_image.paste(image.transpose(Image.FLIP_LEFT_RIGHT), (width * 2, height))
    extended_image.paste(image.transpose(Image.FLIP_TOP_BOTTOM), (width, 0))
    extended_image.paste(image.transpose(Image.FLIP_TOP_BOTTOM), (width, height * 2))
    extended_image.paste(image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM), (0, 0))
    extended_image.paste(image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM), (width * 2, 0))
    extended_image.paste(image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM), (0, height * 2))
    extended_image.paste(image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM), (width * 2, height * 2))

    rotated_image = extended_image.rotate(angle, resample=Image.BICUBIC, center=(width * 3 // 2, height * 3 // 2))
    rotated_image = rotated_image.crop((width, height, width * 2, height * 2))

    return rotated_image


def rotate_image_randomly_symmetry(image, max_angle=20):
    """
    Rotates an image randomly while maintaining symmetry by extending it through mirroring.

    Args:
        image (PIL.Image): Input image.
        max_angle (int, optional): Maximum rotation angle in degrees. Defaults to 20.

    Returns:
        tuple: (Rotated PIL.Image, int angle used for rotation)
    """
    angle = random.randint(-max_angle, max_angle)
    rotated_image = rotate_image_symmetry(image, angle)

    return rotated_image, angle