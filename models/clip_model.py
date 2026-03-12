"""
CLIP model utilities for ShopLens
"""

from transformers import CLIPProcessor, CLIPModel
import torch


def load_clip_model():
    """
    Load the pretrained CLIP model and processor
    """

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    return model, processor


def generate_image_embedding(image, model, processor):
    """
    Generate embedding vector for an image using CLIP
    """

    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")

    # Disable gradient calculation
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        
        # Handle recent transformers versions returning BaseModelOutputWithPooling
        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output

    # Convert tensor → numpy
    embedding = image_features.cpu().numpy()[0]

    return embedding