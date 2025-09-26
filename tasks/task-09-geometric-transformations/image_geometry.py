# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:
    h, w = img.shape
    
    # translated
    shift_y, shift_x = 20, 20
    translated = np.zeros_like(img)
    translated[shift_y:, shift_x:] = img[:h-shift_y, :w-shift_x]
    
    # rotated
    rotated = np.flipud(img.T)  # transpose + flip vertically
    
    # stretched
    new_w = int(w * 1.5)
    x_indices = (np.arange(new_w) / 1.5).astype(int)
    x_indices = np.clip(x_indices, 0, w-1)
    stretched = img[:, x_indices]
    
    # mirror
    mirrored = img[:, ::-1]
    
    y, x = np.indices((h, w))
    x_norm = (2*x - w) / w
    y_norm = (2*y - h) / h
    r = np.sqrt(x_norm**2 + y_norm**2)
    
    k = -0.3
    r_distorted = r * (1 + k * r**2)
    

    r_safe = np.where(r == 0, 1, r)
    x_dist = x_norm * r_distorted / r_safe
    y_dist = y_norm * r_distorted / r_safe
    

    x_mapped = ((x_dist + 1) * w / 2).astype(int)
    y_mapped = ((y_dist + 1) * h / 2).astype(int)
    
    distorted = np.zeros_like(img)
    mask = (x_mapped >= 0) & (x_mapped < w) & (y_mapped >= 0) & (y_mapped < h)
    distorted[y[mask], x[mask]] = img[y_mapped[mask], x_mapped[mask]]
    
    return {
        "translated": translated,
        "rotated": rotated,
        "stretched": stretched,
        "mirrored": mirrored,
        "distorted": distorted
    }