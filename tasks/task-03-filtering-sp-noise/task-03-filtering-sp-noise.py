import cv
import numpy as np

def remove_salt_and_pepper_noise(image: np.ndarray) -> np.ndarray:
    """
    Removes salt and pepper noise from a grayscale image.

    Parameters:
        image (np.ndarray): Noisy input image (grayscale).

    Returns:
        np.ndarray: Denoised image.
    """
    denoised_image = cv.medianBlur(image, 5)
    denoised_image = cv.medianBlur(denoised_image, 3)
    return denoised_image  

if __name__ == "__main__":
    noisy_image = cv.imread("img/head.png", cv.IMREAD_GRAYSCALE)
    denoised_image = remove_salt_and_pepper_noise(noisy_image)
    cv.imwrite("denoised_image.png", denoised_image)
