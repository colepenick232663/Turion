import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from scipy.signal import convolve2d
from skimage.measure import label, regionprops
from mpl_toolkits.mplot3d import Axes3D

def find_brightest_region(image, threshold=200):
    # Convert to grayscale if the image has multiple channels
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Threshold the image to find bright pixels
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Label connected components
    labeled = label(binary)
    regions = regionprops(labeled)
    
    # Find the largest bright region
    if not regions:
        return np.zeros_like(gray), (0, 0)
    
    largest_region = max(regions, key=lambda r: r.area)
    mask = np.zeros_like(gray)
    mask[labeled == largest_region.label] = 255
    
    return mask, largest_region.centroid

def refine_roi(image, centroid, size=50):
    y, x = int(centroid[0]), int(centroid[1])
    h, w = image.shape[:2]
    
    y1, y2 = max(0, y - size), min(h, y + size)
    x1, x2 = max(0, x - size), min(w, x + size)
    
    return image[y1:y2, x1:x2]

def compute_psf(image):
    return image / np.sum(image)

def compute_mtf(image):
    # Compute the Fourier Transform of the image
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    
    # Compute the MTF as the radial average
    cy, cx = np.array(image.shape) // 2
    y, x = np.indices(image.shape)
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    r_bins = np.arange(0, np.max(r), 1)
    mtf = np.array([magnitude_spectrum[(r >= r_bins[i]) & (r < r_bins[i+1])].mean()
                     if ((r >= r_bins[i]) & (r < r_bins[i+1])).any() else 0
                     for i in range(len(r_bins)-1)])
    
    mtf /= mtf.max()
    return mtf, np.linspace(0, 0.5, len(mtf))

def apply_mtf(image, psf):
    return convolve2d(image, psf, mode='same', boundary='wrap')

def compute_contrast(image):
    I_max = np.max(image)
    I_min = np.min(image)
    return (I_max - I_min) / (I_max + I_min)

def process_images(image_path1, image_path2):
    image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    mask, centroid = find_brightest_region(image1)
    refined_region = refine_roi(mask, centroid)
    psf = compute_psf(refined_region)
    mtf_curve, freq = compute_mtf(refined_region)
    
    image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    convolved_image = apply_mtf(image2, psf)
    
    # Compute contrast improvement
    contrast_original = compute_contrast(image2)
    contrast_convolved = compute_contrast(convolved_image)
    contrast_improvement = ((contrast_convolved - contrast_original) / contrast_original) * 100
    
    # Display results
    fig = plt.figure(figsize=(15, 10))
    
    # 3D plot of PSF
    ax1 = fig.add_subplot(231, projection='3d')
    x = np.arange(psf.shape[1])
    y = np.arange(psf.shape[0])
    X, Y = np.meshgrid(x, y)
    ax1.plot_surface(X, Y, psf, cmap='viridis')
    ax1.set_title('Point Spread Function (3D)')
    
    # MTF Curve
    ax2 = fig.add_subplot(232)
    ax2.plot(freq, mtf_curve)
    ax2.set_xlabel('Spatial Frequency (cycles/pixel)')
    ax2.set_ylabel('Contrast')
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, 0.5)
    ax2.set_title('MTF Curve')
    
    # Original second image
    ax3 = fig.add_subplot(233)
    ax3.imshow(image2, cmap='gray')
    ax3.set_title('Original Second Image')
    
    # Convolved second image
    ax4 = fig.add_subplot(234)
    ax4.imshow(convolved_image, cmap='gray')
    ax4.set_title('Convolved Second Image')
    
    # Overlay of original and convolved images
    ax5 = fig.add_subplot(235)
    ax5.imshow(image2, cmap='gray', alpha=0.5)
    ax5.imshow(convolved_image, cmap='jet', alpha=0.5)
    ax5.set_title('Overlay of Original and Convolved Image')
    
    # Display contrast improvement
    print(f"Contrast Improvement: {contrast_improvement:.2f}%")
    
    plt.show()
    
    return convolved_image

# Example usage
if __name__ == "__main__":
    image_path1 = r"C:\Users\cpeni\OneDrive\Pictures\Turion\metadata\alpha centauri useful images\0_1_A2_S13.png"  # Replace with first image path
    image_path2 = r"C:\Users\cpeni\OneDrive\Pictures\Turion\metadata\moon.png"  # Replace with second image path
    process_images(image_path1, image_path2)