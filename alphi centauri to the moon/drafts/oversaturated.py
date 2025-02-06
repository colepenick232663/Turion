import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.restoration import wiener

def find_brightest_spot(image):
    """
    Finds the brightest spot in the image and returns its coordinates.
    """
    brightest_point = np.unravel_index(np.argmax(image), image.shape)
    return brightest_point

def extract_roi(image, center, roi_size=10):
    """
    Extracts a region of interest (ROI) from the image centered at the brightest point.
    The size of the ROI is determined by roi_size.
    """
    y, x = center
    y_min, y_max = max(y - roi_size, 0), min(y + roi_size, image.shape[0])
    x_min, x_max = max(x - roi_size, 0), min(x + roi_size, image.shape[1])
    roi = image[y_min:y_max, x_min:x_max]
    return roi

def resize_roi_to_target(roi, target_shape):
    """
    Resizes the ROI to match the target image shape.
    """
    return cv2.resize(roi, (target_shape[1], target_shape[0]))

def compute_mtf_from_roi(roi):
    """
    Computes the Modulation Transfer Function (MTF) from the ROI.
    For simplicity, we use the normalized Fourier transform of the ROI as the MTF.
    """
    roi_fft = np.fft.fftshift(np.fft.fft2(roi))
    mtf = np.abs(roi_fft)
    mtf /= np.max(mtf)  # Normalize the MTF
    return mtf

def enhance_contrast(image):
    """
    Enhance the contrast of the image using Histogram Equalization.
    """
    return cv2.equalizeHist(image)

def sharpen_image(image):
    """
    Sharpens the image using a kernel to emphasize edges.
    """
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])  # Simple sharpening kernel
    return cv2.filter2D(image, -1, kernel)

def main(image1_path, image2_path):
    # Load the two images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Find the brightest spot in the first image
    brightest_spot = find_brightest_spot(image1)

    # Extract ROI from the first image centered around the brightest spot
    roi = extract_roi(image1, brightest_spot, roi_size=10)

    # Resize the ROI to match the shape of the second image
    roi_resized = resize_roi_to_target(roi, image2.shape)

    # Compute the MTF from the resized ROI
    mtf = compute_mtf_from_roi(roi_resized)

    # Enhance contrast of the second image
    enhanced_contrast_image = enhance_contrast(image2)

    # Apply sharpening to the enhanced contrast image
    sharpened_image = sharpen_image(enhanced_contrast_image)

    # Display the original and sharpened images
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image2, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(sharpened_image, cmap='gray')
    plt.title('Sharpened Image with Contrast')
    
    plt.show()

# Example usage:
image1_path = r"C:\Users\cpeni\OneDrive\Pictures\Turion\metadata\alpha centauri useful images\0_1_A2_S13.png"  # Replace with path to first image (used for ROI)
image2_path = r"C:\Users\cpeni\OneDrive\Pictures\Turion\metadata\moon.png"  # Replace with path to second image (target image)
main(image1_path, image2_path)