"""
Image preprocessing utilities for ShopLens

This file contains helper functions used for
preparing images before sending them to AI models.
"""

import numpy as np
import cv2


def pil_to_numpy(pil_image):
    """
    Convert a PIL image to a NumPy array.

    Parameters:
        pil_image : PIL.Image
            Image uploaded by the user

    Returns:
        numpy_image : numpy.ndarray
            Image converted to NumPy format
    """

    # Convert PIL image to NumPy array
    numpy_image = np.array(pil_image)

    return numpy_image


def resize_image(image, size=(224, 224)):
    """
    Resize image to a fixed size.

    Parameters:
        image : numpy.ndarray
            Image in NumPy format

        size : tuple
            Target size (width, height)

    Returns:
        resized_image : numpy.ndarray
            Resized image
    """

    # Resize image using OpenCV
    resized_image = cv2.resize(image, size)

    return resized_image


def normalize_image(image):
    """
    Normalize image pixel values.

    Converts pixel range from:
    0-255  →  0-1

    Parameters:
        image : numpy.ndarray

    Returns:
        normalized_image : numpy.ndarray
    """

    # Convert pixel values to float
    image = image.astype("float32")

    # Scale pixel values
    normalized_image = image / 255.0

    return normalized_image