import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift
from scipy.integrate import simpson

def find_brightest_spot(image):
    """Find the brightest spot in the image."""
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image)
    return max_loc  # (x, y) coordinates of the brightest spot

def extract_roi(image, center, roi_size):
    """Extract a Region of Interest (ROI) around the center."""
    x, y = center
    half_size = roi_size // 2
    roi = image[y - half_size:y + half_size + 1, x - half_size:x + half_size + 1]
    return roi

def compute_mtf(roi):
    """Compute the MTF from the ROI."""
    roi_normalized = roi / np.sum(roi)  # Normalize the ROI
    mtf = np.abs(fftshift(fft2(roi_normalized)))
    return mtf

def calculate_mtf_curve(mtf):
    """Calculate the MTF curve (contrast ratio vs cycles per pixel)."""
    rows, cols = mtf.shape
    cy = np.arange(rows) - rows // 2
    cx = np.arange(cols) - cols // 2
    fx, fy = np.meshgrid(cx / cols, cy / rows)
    freq = np.sqrt(fx**2 + fy**2)
    freq_normalized = freq / np.max(freq) * 0.5  # Normalize to cycles per pixel (0 to 0.5)

    # Radial average of the MTF
    radial_avg = []
    freq_bins = np.linspace(0, 0.5, 100)
    for f in freq_bins:
        mask = (freq_normalized >= f - 0.01) & (freq_normalized <= f + 0.01)
        if np.any(mask):
            radial_avg.append(np.mean(mtf[mask]))
        else:
            radial_avg.append(0)

    return freq_bins, np.array(radial_avg)

def maximize_mtf_area(image, center, min_roi_size, max_roi_size, step=10):
    """Maximize the area under the MTF curve by adjusting the ROI size."""
    best_area = 0
    best_roi_size = min_roi_size
    best_mtf_curve = None

    for roi_size in range(min_roi_size, max_roi_size + 1, step):
        roi = extract_roi(image, center, roi_size)
        mtf = compute_mtf(roi)
        freq_bins, mtf_curve = calculate_mtf_curve(mtf)
        area = simpson(mtf_curve, freq_bins)  # Calculate area under the curve

        if area > best_area:
            best_area = area
            best_roi_size = roi_size
            best_mtf_curve = (freq_bins, mtf_curve)

    return best_roi_size, best_mtf_curve

def main():
    # Load the uploaded image
    image_path = r"C:\Users\cpeni\OneDrive\Pictures\Turion\metadata\alpha centauri useful images\0_1_A2_S13.png"  # Replace with your image path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        return

    # Find the brightest spot
    brightest_spot = find_brightest_spot(image)
    print(f"Brightest spot coordinates: {brightest_spot}")

    # Maximize the area under the MTF curve by adjusting the ROI size
    min_roi_size = 50
    max_roi_size = 200
    best_roi_size, best_mtf_curve = maximize_mtf_area(image, brightest_spot, min_roi_size, max_roi_size)
    print(f"Best ROI size: {best_roi_size}")

    # Display the MTF curve
    freq_bins, mtf_curve = best_mtf_curve
    plt.figure(figsize=(8, 6))
    plt.plot(freq_bins, mtf_curve, label="MTF Curve")
    plt.xlabel("Cycles per Pixel (0 to 0.5)")
    plt.ylabel("Contrast Ratio")
    plt.title("Modulation Transfer Function (MTF)")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()