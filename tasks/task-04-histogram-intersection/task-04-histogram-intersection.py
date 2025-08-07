import numpy as np

def compute_histogram_intersection(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute the histogram intersection similarity score between two grayscale images.

    This function calculates the similarity between the grayscale intensity 
    distributions of two images by computing the intersection of their 
    normalized 256-bin histograms.

    The histogram intersection is defined as the sum of the minimum values 
    in each corresponding bin of the two normalized histograms. The result 
    ranges from 0.0 (no overlap) to 1.0 (identical histograms).

    Parameters:
        img1 (np.ndarray): First input image as a 2D NumPy array (grayscale).
        img2 (np.ndarray): Second input image as a 2D NumPy array (grayscale).

    Returns:
        float: Histogram intersection score in the range [0.0, 1.0].

    Raises:
        ValueError: If either input is not a 2D array (i.e., not grayscale).
    """    
    if img1.ndim != 2 or img2.ndim != 2:
        raise ValueError("Both input images must be 2D grayscale arrays.")

    ### START CODE HERE ###
    histograma1, bins1 = np.histogram(img1.flatten(), bins=256, range=(0,256))
    histograma2, bins2 = np.histogram(img2.flatten(), bins=256, range=(0,256))

    histograma1_normalizado = histograma1 / np.sum(histograma1)
    histograma2_normalizado = histograma2 / np.sum(histograma2)

    intersection = np.sum(np.minimum(histograma1_normalizado, histograma2_normalizado))
    ### END CODE HERE ###


    return float(intersection)
