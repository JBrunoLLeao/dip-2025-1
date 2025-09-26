"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np
import scikitimage as ski

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    matched_img = np.zeros_like(source_img, dtype=np.uint8)

    for ch in range(3):
        src = source_img[:, :, ch].flatten()
        ref = reference_img[:, :, ch].flatten()

        # histograma
        src_hist, _ = np.histogram(src, bins=256, range=(0, 256))
        ref_hist, _ = np.histogram(ref, bins=256, range=(0, 256))

        # disbtribuição acumulada para matching de histograma
        src_cdf = np.cumsum(src_hist).astype(np.float64)
        ref_cdf = np.cumsum(ref_hist).astype(np.float64)

        # normalização para independência de tamanho da imagem
        src_cdf /= src_cdf[-1]
        ref_cdf /= ref_cdf[-1]

        # mapeamento
        mapping = np.zeros(256, dtype=np.uint8)
        ref_idx = 0
        for src_idx in range(256):
            while ref_idx < 255 and ref_cdf[ref_idx] < src_cdf[src_idx]:
                ref_idx += 1
            mapping[src_idx] = ref_idx

        
        matched_channel = mapping[source_img[:, :, ch]]
        matched_img[:, :, ch] = matched_channel

    return matched_img

