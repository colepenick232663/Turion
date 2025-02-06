import numpy as np
import cv2
import matplotlib.pyplot as plt

def find_brightest_spot(image):
    """
    Finds the brightest spot in the image and returns its coordinates.
    """
    brightest_point = np.unravel_index(np.argmax(image), image.shape)
    return brightest_point

def extract_roi(image, center, roi_size=25):
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

def compute_mtf(roi):
    """
    Compute the MTF from the Fourier transform of the ROI.
    """
    # Compute the FFT of the ROI and take the absolute value (Magnitude)
    roi_fft = np.fft.fftshift(np.fft.fft2(roi))
    mtf = np.abs(roi_fft)
    mtf /= np.max(mtf)  # Normalize the MTF to the max value
    return mtf

def compute_mtf_contrast_ratio(mtf):
    """
    Compute the contrast ratio as a function of cycles per pixel (0 to 0.5).
    """
    # Sum MTF values along both axes to reduce the MTF to 1D for contrast calculation
    mtf_sum = np.sum(mtf, axis=0)  # Sum across rows (for contrast profile)
    
    # Define cycles per pixel as the frequency range from 0 to 0.5
    cycs_per_pixel = np.fft.fftfreq(mtf.shape[1], d=1.0)[:mtf.shape[1]//2]  # Frequencies in cycles per pixel
    
    # Normalize the contrast ratio to get the contrast profile
    contrast_ratio = mtf_sum[:mtf.shape[1]//2]  # Take only the positive frequencies (0 to 0.5)
    
    return cycs_per_pixel, contrast_ratio

def main(image1_path, image2_path):
    # Load the two images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Find the brightest spot in the first image
    brightest_spot = find_brightest_spot(image1)

    # Extract ROI from the first image centered around the brightest spot (PSF)
    roi = extract_roi(image1, brightest_spot, roi_size=50)  # Increase ROI size to 50 pixels

    # Resize the ROI to match the shape of the second image
    roi_resized = resize_roi_to_target(roi, image2.shape)

    # Sharpen the second image
    sharpened_image = sharpen_image(image2)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to avoid overexposure
    enhanced_contrast_image = adjust_contrast(sharpened_image)

    # Compute the MTF from the ROI
    mtf = compute_mtf(roi_resized)

    # Compute the contrast ratio as a function of cycles per pixel
    cycs_per_pixel, contrast_ratio = compute_mtf_contrast_ratio(mtf)

    # Normalize contrast ratio so that it peaks at 1.0 at 0 cycles per pixel
    contrast_ratio = contrast_ratio / np.max(contrast_ratio)  # Normalize to 1.0 at 0 cycles per pixel

    # Display the original, PSF ROI, and enhanced images
    plt.figure(figsize=(12, 6))
    
    # Original Image
    plt.subplot(2, 3, 1)
    plt.imshow(image2, cmap='gray')
    plt.title('Original Image')
    
    # Sharpened & Contrast Enhanced Image
    plt.subplot(2, 3, 2)
    plt.imshow(enhanced_contrast_image, cmap='gray')
    plt.title('Sharpened & Contrast Enhanced Image')

    # PSF ROI from Image 1
    plt.subplot(2, 3, 3)
    plt.imshow(roi, cmap='gray')
    plt.title('PSF ROI from Image 1')

    # Plot the MTF contrast ratio vs cycles per pixel
    plt.subplot(2, 3, 4)
    plt.plot(cycs_per_pixel, contrast_ratio)
    plt.title('MTF Contrast Ratio vs Cycles per Pixel')
    plt.xlabel('Cycles per Pixel')
    plt.ylabel('Contrast Ratio')

    plt.tight_layout()
    plt.show()

# Example usage:
image1_path = r"C:\Users\cpeni\OneDrive\Pictures\Turion\metadata\alpha centauri useful images\0_1_A2_S13.png"  # Replace with path to first image (used for ROI)
image2_path = r"C:\Users\cpeni\OneDrive\Pictures\Turion\metadata\moon.png"  # Replace with path to second image (target image)
main(image1_path, image2_path)