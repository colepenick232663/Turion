import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def find_brightest_spot(image):
    """
    Finds the brightest spot in the image and returns its coordinates.
    """
    return np.unravel_index(np.argmax(image), image.shape)

def extract_roi(image, center, roi_size=50):
    """
    Extracts a region of interest (ROI) from the image centered at the brightest point.
    """
    y, x = center
    y_min, y_max = max(y - roi_size, 0), min(y + roi_size, image.shape[0])
    x_min, x_max = max(x - roi_size, 0), min(x + roi_size, image.shape[1])
    return image[y_min:y_max, x_min:x_max]

def filter_dense_bright_region(roi):
    """
    Zeroes out pixels that aren't part of the main dense bright region in the PSF.
    """
    _, thresholded = cv2.threshold(roi, np.max(roi) * 0.5, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresholded.astype(np.uint8))
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8)
    else:
        mask = np.ones_like(roi, dtype=np.uint8)
    return roi * mask, mask

def compute_mtf(filtered_roi):
    """
    Compute the MTF from the Fourier transform of the filtered PSF.
    """
    roi_fft = np.fft.fft2(filtered_roi)
    roi_fft_magnitude = np.abs(np.fft.fftshift(roi_fft))
    mtf = roi_fft_magnitude / np.max(roi_fft_magnitude)
    return mtf

def compute_mtf_curve(mtf):
    """
    Compute the contrast ratio as a function of normalized cycles per pixel.
    """
    mtf_sum = np.sum(mtf, axis=0)  # Sum across rows for 1D profile
    nyquist_index = mtf.shape[1] // 2
    freq_axis = np.fft.fftfreq(mtf.shape[1])[:nyquist_index]  # Take only positive frequencies
    freq_axis = np.abs(freq_axis)  # Ensure frequencies are sorted from 0 to Nyquist
    contrast_ratio = mtf_sum[:nyquist_index]
    contrast_ratio /= np.max(contrast_ratio)
    contrast_ratio = contrast_ratio[::-1]  # Mirror the MTF curve
    sorted_indices = np.argsort(freq_axis)  # Sort indices to ensure increasing order
    
    # Smooth the MTF curve using interpolation
    interp_func = interp1d(freq_axis[sorted_indices], contrast_ratio[sorted_indices], kind='cubic')
    smooth_freq_axis = np.linspace(freq_axis[sorted_indices][0], freq_axis[sorted_indices][-1], 500)
    smooth_contrast_ratio = interp_func(smooth_freq_axis)
    
    return smooth_freq_axis, smooth_contrast_ratio

def main(image1_path, image2_path):
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    brightest_spot = find_brightest_spot(image1)
    roi = extract_roi(image1, brightest_spot, roi_size=50)
    filtered_roi, mask = filter_dense_bright_region(roi)
    mtf = compute_mtf(filtered_roi)
    freq_axis, contrast_ratio = compute_mtf_curve(mtf)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image1, cmap='gray')
    plt.title('Original Image 1')
    plt.subplot(1, 3, 2)
    plt.imshow(roi, cmap='gray')
    plt.title('Extracted ROI')
    plt.subplot(1, 3, 3)
    plt.imshow(filtered_roi, cmap='gray')
    plt.title('Filtered PSF Region')
    plt.figure()
    plt.plot(freq_axis, contrast_ratio, marker='o')
    plt.title('Smoothed MTF Curve from Filtered PSF')
    plt.xlabel('Normalized Cycles per Pixel')
    plt.ylabel('Contrast Ratio')
    plt.grid(True)
    plt.show()

image1_path = r"C:\Users\cpeni\OneDrive\Pictures\Turion\metadata\alpha centauri useful images\0_1_A2_S13.png"
image2_path = r"C:\Users\cpeni\OneDrive\Pictures\Turion\metadata\moon.png"
main(image1_path, image2_path)