import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
from tkinter import Tk, filedialog

# Function to upload image
def upload_image():
    Tk().withdraw()  # Hide Tkinter root window
    file_path = filedialog.askopenfilename()
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

# Gaussian function for PSF fitting
def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, offset):
    x, y = xy
    g = offset + amplitude * np.exp(-(((x - xo) ** 2) / (2 * sigma_x ** 2) + ((y - yo) ** 2) / (2 * sigma_y ** 2)))
    return g.ravel()

# Detect bright blobs and compute Gaussian PSF
def compute_psf(image):
    _, binary = cv2.threshold(image, np.max(image) * 0.8, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("No bright blobs found.")
        return None, None, None
    
    # Find largest bright blob
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    blob = image[y:y+h, x:x+w]
    
    # Fit Gaussian PSF
    X, Y = np.meshgrid(np.arange(blob.shape[1]), np.arange(blob.shape[0]))
    initial_guess = (np.max(blob), w/2, h/2, w/4, h/4, np.min(blob))
    popt, _ = curve_fit(gaussian_2d, (X, Y), blob.ravel(), p0=initial_guess)
    
    return blob, popt, (X, Y)

# Compute MTF from PSF
def compute_mtf(psf):
    psf_fft = np.abs(fft(psf, axis=1))
    mtf = psf_fft[:, :psf.shape[1] // 2]
    freq = fftfreq(psf.shape[1])[:psf.shape[1] // 2]
    
    return freq, np.mean(mtf, axis=0)

# Main script
if __name__ == "__main__":
    image = upload_image()
    if image is None:
        print("No image selected.")
    else:
        psf, popt, (X, Y) = compute_psf(image)
        if psf is not None:
            freq, mtf = compute_mtf(psf)
            
            # Plot PSF
            plt.figure(figsize=(8, 5))
            plt.imshow(psf, cmap='gray')
            plt.title("Extracted PSF")
            plt.colorbar()
            plt.show()
            
            # Plot MTF
            plt.figure(figsize=(8, 5))
            plt.plot(freq, mtf, label='MTF')
            plt.xlabel('Spatial Frequency')
            plt.ylabel('MTF')
            plt.title('Modulation Transfer Function')
            plt.legend()
            plt.grid()
            plt.show()
