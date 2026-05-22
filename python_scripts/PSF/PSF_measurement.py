# %% Imports and functions

import napari
import numpy as np
from tifffile import TiffFile
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# -------------------------------
# GAUSSIAN FIT FUNCTIONS
# -------------------------------
def get_pixel_size_um(pixel_size_sensor, magnification):
    return pixel_size_sensor / magnification

def gaussian_1d(x, amp, x0, sigma, offset):
    # 1D Gaussian function
    # amp - amplitude
    # x0, y0 - center
    # sigma_x, sigma_y - x and y spreads (standard deviations)
    # offset - background intensity
    return offset + amp * np.exp(-((x - x0)**2) / (2 * sigma**2))

def sum_roi_along_axis(roi, axis=1):
    # Sums the ROI along the specified axis (0 for y, 1 for x)
    roi_sum =  np.sum(roi, axis=axis)
    return roi_sum

def fit_gaussian_to_roi_sum(roi_sum):
    # Fits 1D Gaussian to the summed ROI and returns FWHM
    x_size = roi_sum.shape[0]
    x = np.arange(x_size)
    initial_guess = (roi_sum.max(), x_size/2, x_size/4, roi_sum.min()) #Initial spread guess of x_size/4 pixels
    params, _ = curve_fit(gaussian_1d, x, roi_sum, p0=initial_guess)
    amp, x0, sigma, offset = params
    fwhm = 2.355 * sigma #Width of the Gaussian at half maximum
    return fwhm, params

def plot_roi_sum_and_fit(roi_sum, params):
    x_size = roi_sum.shape[0]
    x = np.arange(x_size)
    fit_data = gaussian_1d(x, *params)

    plt.figure(figsize=(6, 4))
    plt.title("ROI Sum and Gaussian Fit")
    plt.plot(x, roi_sum, label='ROI Sum', color='blue')
    plt.plot(x, fit_data, label='Gaussian Fit', color='red', linestyle='--')
    plt.xlabel('Pixel Position')
    plt.ylabel('Summed Intensity along x axis')
    plt.legend()
    plt.tight_layout()
    plt.show()

def rayleigh_criterion(lambda_um, NA):
    # Rayleigh criterion: how close two points can be to be resolved
    return 0.61 * lambda_um / NA

def sparrow_limit(lambda_um, NA):
    # Sparrow limit: how close two points can be to be resolved
    return 0.47 * lambda_um / NA

def abbe_limit(lambda_um, NA):
    # Lateral resolution formula - smallest resolvable feature size
    return 0.5 * lambda_um / NA

# %% Initialization
# -------------------------------
# PARAMETERS
# -------------------------------
pixel_size_sensor = 6.5  # µm
magnification = 50
pixel_size_um = get_pixel_size_um(pixel_size_sensor, magnification)

lambda_um = 0.580  # µm (emission wavelength)
NA = 0.6
# theoretical_lateral_resolution = lateral_resolution(lambda_um, NA)
theoretical_rayleigh_resolution = rayleigh_criterion(lambda_um, NA)
abbe_limit_val = abbe_limit(lambda_um, NA)
print(f"Abbe limit: {abbe_limit_val:.3f} µm")
print(f"Theoretical Rayleigh resolution: {theoretical_rayleigh_resolution:.3f} µm")
print(f"Sparrow limit: {sparrow_limit(lambda_um, NA):.3f} µm")

# -------------------------------
# LOAD IMAGE
# -------------------------------
file_num = 1
# 1 - multiple beads in focus
# 2 - one bead very much in focus, others not so much
# 3 - out of focus bead
# 4 - looks like 2 beads next to each other, hard to distinguish
# 5 - looks like two beads i could use
# 6 - 1 aggregate in focus, others out of focus
file_path = Path(rf"C:\Users\labudzki\OneDrive - AMOLF\Documents\Repositories\confocal_processing\PSF runs\Run{file_num}\Run{file_num}_MMStack_Pos0.ome.tif")

with TiffFile(file_path) as tif:
    stack = tif.asarray()

stack = np.array(stack, dtype='float32')
print(f"Image shape: {stack.shape}")

# -------------------------------
# %% NAPARI VIEWER
# -------------------------------
viewer = napari.Viewer()
viewer.add_image(stack, name='Raw img', multiscale=False, axis_labels=['Y','X'], contrast_limits=[16, 780])

# Add Shapes layer for ROI selection
shapes_layer = viewer.add_shapes(name='ROI', shape_type='rectangle', edge_color='red', face_color='transparent')

@shapes_layer.events.data.connect
def on_roi_added(event):
    if len(shapes_layer.data) > 0:
        roi_coords = shapes_layer.data[-1]
        y_min, y_max = int(min(roi_coords[:, 0])), int(max(roi_coords[:, 0]))
        x_min, x_max = int(min(roi_coords[:, 1])), int(max(roi_coords[:, 1]))
        roi = stack[y_min:y_max, x_min:x_max]
        print(f"ROI shape: {roi.shape}")

        # Add cropped ROI as new layer
        viewer.add_image(roi, name='Cropped ROI')

        # Fit Gaussian
        roi_sum = sum_roi_along_axis(roi, axis=1)
        fwhm_x, params = fit_gaussian_to_roi_sum(roi_sum)
        # fwhm_x, fwhm_y, params = fit_gaussian_to_roi_sum(roi)
        print(f"FWHM X: {fwhm_x:.2f} px ({fwhm_x * pixel_size_um:.3f} µm)")
        # print(f"FWHM Y: {fwhm_y:.2f} px ({fwhm_y * pixel_size_um:.3f} µm)")

        # Plot ROI and Gaussian fit overlay
        plot_roi_sum_and_fit(roi_sum, params)

# GUI runs and blocks until closed
napari.run()

# -------------------------------
# %% Processing ROI data after viewer is closed
# -------------------------------

# This runs after the Napari window(s) are closed
# copy ROI data so we don't rely on the viewer being alive
rois = [np.array(s) for s in shapes_layer.data]
print("Number of ROIs drawn:", len(rois))
for i, roi in enumerate(rois):
    y_min, y_max = int(min(roi[:,0])), int(max(roi[:,0]))
    x_min, x_max = int(min(roi[:,1])), int(max(roi[:,1]))
    # print(f"ROI {i}: shape {(y_max-y_min, x_max-x_min)}")
    # further processing here...


# -------------------------------
# fwhm x: 5.45 px (0.708 um)
# bead size is 0.02 um 
# figure out a way to call this code ^^ in a new file so that i can call multiple times and save the data, and make some statistics
# compare to theoretical lateral resolution