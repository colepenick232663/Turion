import numpy as np
import cv2
import matplotlib.pyplot as plt

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

def sharpen_image(image):
    """
    Sharpens the image using a kernel to emphasize edges.
    """
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])  # Simple sharpening kernel
    return cv2.filter2D(image, -1, kernel)

def adjust_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance the contrast of the image
    while limiting overexposure in bright regions.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

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

    # Sharpen the second image
    sharpened_image = sharpen_image(image2)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to avoid overexposure
    enhanced_contrast_image = adjust_contrast(sharpened_image)

    # Display the original and the sharpened + contrast-enhanced images
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image2, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_contrast_image, cmap='gray')
    plt.title('Sharpened & Contrast Enhanced Image')
    
    plt.show()

# Example usage:
image1_path = r"C:\Users\cpeni\OneDrive\Pictures\Turion\metadata\alpha centauri useful images\0_1_A2_S13.png"  # Replace with path to first image (used for ROI)
image2_path = r"C:\Users\cpeni\OneDrive\Pictures\Turion\metadata\moon.png"  # Replace with path to second image (target image)
main(image1_path, image2_path)