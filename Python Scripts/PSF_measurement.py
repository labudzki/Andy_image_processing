# # import napari
# # import numpy as np
# # from tifffile import TiffFile
# # from pathlib import Path
# # from LOCOMYCO_ImgAnalysis import tifreading as u_tif

# # fnum = 3

# # path_movie = Path(
# # rf"C:\Users\labudzki\OneDrive - AMOLF\Documents\Repositories\confocal_processing\PSF runs\Run2\Pic2_MMStack_Pos0.ome.tif"
# # )

# # # Load as a NumPy array (ensure it’s contiguous)
# # with TiffFile(path_movie) as tif:
# #     stack = tif.asarray()
    

# # #stack=u_tif.load_tiff_stack(path_movie, squeeze=True)

# # stack = np.array(stack, dtype='float32')  # reduce memory usage

# # print(stack.shape, stack.dtype)  # sanity check


# # viewer = napari.Viewer()

# # viewer.add_image(stack, name='Raw img', multiscale=False, axis_labels=['Y','X'])
# # # viewer.add_image(stack, name='Raw video', multiscale=False, axis_labels=['Z','Y','X']) # for Run1 and Run2
# # # viewer.add_image(stack, name='Raw video', multiscale=False, axis_labels=['Time','Y','X']) # for Run5
# # # #print(viewer.layers.data


# # shapes_layer = viewer.add_shapes(
# #     name='Measurements',
# #     shape_type='line',  # You can draw lines for measuring
# #     edge_color='yellow',
# #     face_color='transparent'
# # )


# # for shape in shapes_layer.data:
# #     # Each shape is an array of coordinates
# #     p1, p2 = shape  # For a line, there are two points
# #     distance_pixels = np.linalg.norm(p2 - p1)
# #     print(f"Distance in pixels: {distance_pixels}")

# # napari.run()


# import napari
# import numpy as np
# from tifffile import TiffFile
# from pathlib import Path

# # Calculating true pixel size
# pixel_size_sensor = 6.5 #um
# magnification = 50
# pixel_size_um = pixel_size_sensor / magnification

# # Calculating PSF in image
# bead_size = 0.024 #um
# # PSF = 

# path_movie = Path(r"C:\Users\labudzki\OneDrive - AMOLF\Documents\Repositories\confocal_processing\PSF runs\Run2\Pic2_MMStack_Pos0.ome.tif")

# with TiffFile(path_movie) as tif:
#     stack = tif.asarray()
#     # Extract pixel size if available
#     # ome_meta = tif.ome_metadata
#     # print(ome_meta)

# stack = np.array(stack, dtype='float32')
# print(stack.shape, stack.dtype)

# viewer = napari.Viewer()
# viewer.add_image(stack, name='Raw img', multiscale=False, axis_labels=['Y','X'])

# # Add Shapes layer for measurements
# shapes_layer = viewer.add_shapes(name='Shapes', shape_type='line', edge_color='yellow', face_color='transparent')

# @shapes_layer.events.data.connect
# def on_shape_added(event):
#     for shape in shapes_layer.data:
#         p1, p2 = shape
#         distance_pixels = np.linalg.norm(p2 - p1)
#         print(f"Distance in pixels: {distance_pixels}")
#         # True distance:
#         print(f"Distance: {distance_pixels * pixel_size_um} µm")

# napari.run()

# PSF = distance_pixels * pixel_size_um

# import napari
# import numpy as np
# from tifffile import TiffFile
# from pathlib import Path
# from scipy.optimize import curve_fit

# # -------------------------------
# # PARAMETERS
# # -------------------------------
# pixel_size_sensor = 6.5  # µm (camera pixel size)
# magnification = 50
# pixel_size_um = pixel_size_sensor / magnification  # effective pixel size at sample

# # -------------------------------
# # GAUSSIAN FIT FUNCTIONS
# # -------------------------------
# def gaussian_2d(coords, amp, xo, yo, sigma_x, sigma_y, offset):
#     x, y = coords
#     xo = float(xo)
#     yo = float(yo)
#     g = offset + amp * np.exp(-(((x - xo)**2)/(2*sigma_x**2) + ((y - yo)**2)/(2*sigma_y**2)))
#     return g.ravel()

# def fit_gaussian_to_roi(roi):
#     y_size, x_size = roi.shape
#     x = np.arange(x_size)
#     y = np.arange(y_size)
#     x, y = np.meshgrid(x, y)
#     initial_guess = (roi.max(), x_size/2, y_size/2, 2, 2, roi.min())
#     popt, _ = curve_fit(gaussian_2d, (x, y), roi.ravel(), p0=initial_guess)
#     amp, xo, yo, sigma_x, sigma_y, offset = popt
#     fwhm_x = 2.355 * sigma_x
#     fwhm_y = 2.355 * sigma_y
#     return fwhm_x, fwhm_y, popt

# # -------------------------------
# # LOAD IMAGE
# # -------------------------------
# path_movie = Path(r"C:\Users\labudzki\OneDrive - AMOLF\Documents\Repositories\confocal_processing\PSF runs\Run2\Pic2_MMStack_Pos0.ome.tif")

# with TiffFile(path_movie) as tif:
#     stack = tif.asarray()

# stack = np.array(stack, dtype='float32')
# print(f"Image shape: {stack.shape}, dtype: {stack.dtype}")

# # -------------------------------
# # NAPARI VIEWER
# # -------------------------------
# viewer = napari.Viewer()
# viewer.add_image(stack, name='Raw img', multiscale=False, axis_labels=['Y','X'])

# # Add Shapes layer for ROI selection
# shapes_layer = viewer.add_shapes(name='ROI', shape_type='rectangle', edge_color='red', face_color='transparent')

# @shapes_layer.events.data.connect
# def on_roi_added(event):
#     if len(shapes_layer.data) > 0:
#         roi_coords = shapes_layer.data[-1]  # last drawn rectangle
#         y_min, y_max = int(min(roi_coords[:, 0])), int(max(roi_coords[:, 0]))
#         x_min, x_max = int(min(roi_coords[:, 1])), int(max(roi_coords[:, 1]))
#         roi = stack[y_min:y_max, x_min:x_max]
#         print(f"ROI shape: {roi.shape}")

#         # Fit Gaussian
#         fwhm_x, fwhm_y, params = fit_gaussian_to_roi(roi)
#         print(f"FWHM X: {fwhm_x:.2f} px ({fwhm_x * pixel_size_um:.3f} µm)")
#         print(f"FWHM Y: {fwhm_y:.2f} px ({fwhm_y * pixel_size_um:.3f} µm)")

# napari.run()

import napari
import numpy as np
from tifffile import TiffFile
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# -------------------------------
# PARAMETERS
# -------------------------------
pixel_size_sensor = 6.5  # µm
magnification = 50
pixel_size_um = pixel_size_sensor / magnification

# -------------------------------
# GAUSSIAN FIT FUNCTIONS
# -------------------------------
def gaussian_2d(coords, amp, x0, y0, sigma_x, sigma_y, offset):
    # Gaussian function in 2D
    # amp - amplitude
    # x0, y0 - center
    # sigma_x, sigma_y - x and y spreads (standard deviations)
    # offset - background intensity
    x, y = coords
    x0 = float(x0)
    y0 = float(y0)
    g = offset + amp * np.exp(-(((x - x0)**2)/(2*sigma_x**2) + ((y - y0)**2)/(2*sigma_y**2)))
    return g.ravel() # turns [[1,2],[3,4]] into [1,2,3,4]. used for curve_fit compatibility

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

# def fit_gaussian_to_roi(roi):
#     # Fits 2D Gaussian to the selected ROI and returns FWHM in x and y
#     y_size, x_size = roi.shape
#     x = np.arange(x_size)
#     y = np.arange(y_size)
#     x, y = np.meshgrid(x, y)
#     initial_guess = (roi.max(), x_size/2, y_size/2, 3, 3, roi.min()) #Initial spread guess of 3 pixels in x and y
#     params, _ = curve_fit(gaussian_2d, (x, y), roi.ravel(), p0=initial_guess)
#     amp, x0, y0, sigma_x, sigma_y, offset = params
#     fwhm_x = 2.355 * sigma_x #Width of the Gaussian at half maximum
#     fwhm_y = 2.355 * sigma_y
#     return fwhm_x, fwhm_y, params

def plot_roi_and_fit(roi, params):
    y_size, x_size = roi.shape
    x = np.arange(x_size)
    y = np.arange(y_size)
    x, y = np.meshgrid(x, y)
    fit_data = gaussian_2d((x, y), *params).reshape(roi.shape)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("ROI")
    plt.imshow(roi, cmap='gray')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.title("Gaussian Fit Overlay")
    plt.imshow(roi, cmap='gray')
    plt.contour(fit_data, colors='red', linewidths=1)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

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

def lateral_resolution(lambda_um, NA):
    return 0.61 * lambda_um / NA

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
print(f"Image shape: {stack.shape}, dtype: {stack.dtype}")

# -------------------------------
# NAPARI VIEWER
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

napari.run()


# -------------------------------
# fwhm x: 5.45 px (0.708 um)
# check what the actual size is based on calculation
# figure out a way to call this code ^^ in a new file so that i can call multiple times and save the data, and make some statistics
# compare to theoretical lateral resolution