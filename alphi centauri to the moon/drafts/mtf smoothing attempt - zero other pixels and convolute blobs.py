import cv2
import numpy as np
import scipy.ndimage as ndi
from skimage.filters import gaussian
import matplotlib.pyplot as plt

def find_brightest_regions(image, threshold=200):
    """Find the densest collection of brightest pixels."""
    if len(image.shape) == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = image  # Already grayscale
    bright_pixels = grayscale >= threshold  # Boolean mask for bright regions
    return bright_pixels.astype(np.uint8) * 255  # Convert to 255 scale

def apply_convolution(image, kernel_size=3):
    """Smooth blobs using Gaussian blur to approximate PSF."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def extract_and_scale_roi(image, mask, output_size=(100, 100)):
    """Extract ROI around the brightest blobs and scale it to 100x100 pixels, ensuring visibility."""
    labels, num_features = ndi.label(mask)
    sizes = ndi.sum(mask, labels, range(num_features + 1))
    sorted_indices = np.argsort(sizes)[::-1]  # Sort by size, descending
    
    if len(sorted_indices) > 1:
        idx1, idx2 = sorted_indices[:2]  # Take the two largest blobs
        combined_mask = ((labels == idx1) | (labels == idx2)).astype(np.uint8) * 255
    else:
        combined_mask = (labels == sorted_indices[0]).astype(np.uint8) * 255
    
    x, y, w, h = cv2.boundingRect(combined_mask)
    roi = image[y:y+h, x:x+w]
    return cv2.resize(roi, output_size, interpolation=cv2.INTER_LINEAR)

def compute_psf_mtf(image):
    """Compute MTF from the PSF of the convoluted blobs."""
    if len(image.shape) == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = image  # Already grayscale
    psf = gaussian(grayscale, sigma=1.5)
    psf_ft = np.abs(np.fft.fft2(psf))  # Fourier Transform of PSF
    psf_ft = np.fft.fftshift(psf_ft)  # Shift zero frequency to center
    mtf = psf_ft / np.max(psf_ft)  # Normalize
    freq = np.fft.fftfreq(psf.shape[0])
    freq = np.fft.fftshift(freq)  # Shift frequency axis
    valid_indices = (freq >= 0) & (freq <= 0.5)  # Limit to 0 - 0.5 cycles/pixel
    return freq[valid_indices], np.mean(mtf, axis=1)[valid_indices]

def plot_mtf_curve(freq, mtf_curve):
    """Plot the MTF curve with normalized range."""
    plt.figure(figsize=(6, 4))
    plt.plot(freq, mtf_curve, label="MTF Curve")
    plt.xlabel("Spatial Frequency (cycles per pixel)")
    plt.ylabel("MTF Response (0 to 1)")
    plt.title("MTF Curve from PSF of Convoluted Blobs")
    plt.legend()
    plt.grid()
    plt.show()

def process_image(image_path):
    """Main function to process the image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not open or find the image!")
    
    bright_regions = find_brightest_regions(image)
    smoothed = apply_convolution(bright_regions)
    roi_image = extract_and_scale_roi(smoothed, smoothed)
    
    return roi_image, smoothed, image

def show_images(original, processed, roi):
    """Displays original, processed, and scaled ROI images."""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.title("Processed Image")
    plt.imshow(processed, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.title("Scaled ROI (100x100) - Visible Blobs")
    plt.imshow(roi, cmap="gray")
    plt.axis("off")
    
    plt.show()

if __name__ == "__main__":
    image_path = r"C:\Users\cpeni\OneDrive\Pictures\Turion\metadata\alpha centauri useful images\0_1_A2_S13.png"  # Change to your image file
    roi_image, smoothed_image, original_image = process_image(image_path)
    show_images(original_image, smoothed_image, roi_image)
    cv2.imwrite("output_roi.jpg", roi_image)
    
    # Compute and plot MTF curve from the convoluted blobs
    freq, mtf_curve = compute_psf_mtf(roi_image)
    plot_mtf_curve(freq, mtf_curve)