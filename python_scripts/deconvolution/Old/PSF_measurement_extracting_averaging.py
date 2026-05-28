# %% Imports and functions

import napari
import numpy as np
from tifffile import TiffFile
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.ndimage import shift
import matplotlib.pyplot as plt
from skimage import registration  # only for subpixel alignment

# -------------------------------
# GAUSSIAN FIT FUNCTIONS 
# -------------------------------
def get_pixel_size_um(pixel_size_sensor, magnification):
    return pixel_size_sensor / magnification

def gaussian_1d(x, amp, x0, sigma, offset):
    return offset + amp * np.exp(-((x - x0)**2) / (2 * sigma**2))

def fit_gaussian_to_profile(profile_1d):
    n = profile_1d.shape[0]
    x = np.arange(n)
    # robust-ish initial guess
    amp0 = float(profile_1d.max() - profile_1d.min())
    x0_0 = float(np.argmax(profile_1d))
    sigma0 = max(2.0, n / 6.0)
    off0 = float(profile_1d.min())
    p0 = (amp0, x0_0, sigma0, off0)
    params, _ = curve_fit(gaussian_1d, x, profile_1d, p0=p0, maxfev=20000)
    amp, x0, sigma, offset = params
    fwhm = 2.355 * sigma
    return fwhm, params

def plot_profile_and_fit(profile_1d, params, title="Profile & Gaussian fit", xlabel="Pixel"):
    n = profile_1d.shape[0]
    x = np.arange(n)
    fit_y = gaussian_1d(x, *params)

    plt.figure(figsize=(6,4))
    plt.title(title)
    plt.plot(x, profile_1d, 'b-', label='Profile')
    plt.plot(x, fit_y, 'r--', label='Gaussian fit')
    plt.xlabel(xlabel); plt.ylabel('Summed intensity')
    plt.legend(); plt.tight_layout(); plt.show()

def rayleigh_criterion(lambda_um, NA): return 0.61 * lambda_um / NA
def sparrow_limit(lambda_um, NA): return 0.47 * lambda_um / NA
def abbe_limit(lambda_um, NA): return 0.5 * lambda_um / NA

# -------------------------------
# Utilities for manual single-bead selection
# -------------------------------
def crop_rect(img, rect_vertices):
    """Crop an image using a napari rectangle (4x2 vertices in (row, col))."""
    y_min = int(np.floor(np.min(rect_vertices[:,0])))
    y_max = int(np.ceil (np.max(rect_vertices[:,0])))
    x_min = int(np.floor(np.min(rect_vertices[:,1])))
    x_max = int(np.ceil (np.max(rect_vertices[:,1])))
    return img[y_min:y_max, x_min:x_max], (y_min, x_min)

def recenter_and_extract_fixed(img_crop, origin_rc, bead_window=21):
    """
    From a user-selected crop (assumed to contain ONE bead), find the peak inside,
    then extract a fixed-size bead window centered on that peak in GLOBAL coordinates.
    Returns: bead_roi (bead_window x bead_window), peak_global_rc
    """
    # Find peak inside the selected crop
    local_peak_rc = np.unravel_index(np.argmax(img_crop), img_crop.shape)
    peak_r_global = origin_rc[0] + local_peak_rc[0]
    peak_c_global = origin_rc[1] + local_peak_rc[1]

    half = bead_window // 2
    r1 = peak_r_global - half
    r2 = peak_r_global + half + 1
    c1 = peak_c_global - half
    c2 = peak_c_global + half + 1

    # Bounds check
    H, W = viewer.layers['Raw img'].data.shape
    if r1 < 0 or c1 < 0 or r2 > H or c2 > W:
        return None, (peak_r_global, peak_c_global)

    bead_roi = viewer.layers['Raw img'].data[r1:r2, c1:c2].astype(np.float32)
    return bead_roi, (peak_r_global, peak_c_global)

def normalize_and_align(rois, align_subpixel=True, norm_mode="max", upsample_factor=50):
    if len(rois) == 0:
        return []
    normed = []
    for roi in rois:
        if norm_mode == "sum":
            s = roi.sum()
            normed.append(roi / s if s > 0 else roi.copy())
        else:
            m = roi.max()
            normed.append(roi / m if m > 0 else roi.copy())

    if not align_subpixel or len(normed) == 1:
        return normed

    ref = normed[0]
    aligned = []
    for roi in normed:
        shift_est, _, _ = registration.phase_cross_correlation(ref, roi, upsample_factor=upsample_factor)
        aligned.append(shift(roi, shift=shift_est, mode='nearest', order=1))
    return aligned

def average_psf(rois):
    return None if len(rois) == 0 else np.mean(np.stack(rois, axis=0), axis=0)

def compute_fwhm_xy_from_psf(psf):
    prof_x = psf.sum(axis=0)
    prof_y = psf.sum(axis=1)
    fwhm_x_px, p_x = fit_gaussian_to_profile(prof_x)
    fwhm_y_px, p_y = fit_gaussian_to_profile(prof_y)
    return fwhm_x_px, p_x, fwhm_y_px, p_y, prof_x, prof_y

# %% Initialization
# -------------------------------
# PARAMETERS
pixel_size_sensor = 6.5  # µm
magnification = 60
pixel_size_um = get_pixel_size_um(pixel_size_sensor, magnification)

lambda_um = 0.605  # µm
NA = 0.60
print(f"Abbe limit: {abbe_limit(lambda_um, NA):.3f} µm")
print(f"Rayleigh:   {rayleigh_criterion(lambda_um, NA):.3f} µm")
print(f"Sparrow:    {sparrow_limit(lambda_um, NA):.3f} µm")

# Selection/averaging params
bead_window = 21         # fixed extracted bead ROI size (odd number)
align_subpixel = True
norm_mode = "max"        # 'max' or 'sum'

# -------------------------------
# LOAD IMAGE
# -------------------------------
file_path = Path(r"C:\Users\labudzki\AMOLF-SHIMIZU Dropbox\DATA\Ach_data\x. SetUp Charac\Point spread function\260316\PSF_20x_nanobeads680-605\PSF_20x_nanobeads680-605_MMStack_Pos0_t56.tif")
with TiffFile(file_path) as tif:
    stack = tif.asarray()

# pick first plane of stack
img2d = stack[0].astype('float32') if stack.ndim > 2 else stack.astype('float32')


# -------------------------------
# Storage & registry
# -------------------------------
selected_bead_rois = []                # list[np.ndarray] : fixed-size bead windows
selected_bead_centers_global = []      # list[tuple(r,c)] : centers used
crop_layer_names = []                  # list[str]        : napari layer names for crops (optional)
# Important: these lists share the SAME index order as points_layer.data rows.


# -------------------------------
# NAPARI VIEWER
# -------------------------------
viewer = napari.Viewer()
raw_layer = viewer.add_image(img2d, name='Raw img', axis_labels=['Y','X'])
shapes_layer = viewer.add_shapes(
    name='ROI (one bead each)', shape_type='rectangle',
    edge_color='cyan', face_color='transparent'
)
points_layer = viewer.add_points(name='Bead centers', size=6, face_color='yellow')

def handle_new_rectangle(rect_vertices):
    """Called whenever a rectangle is added. We treat it as ONE bead selection."""
    # Crop using rectangle
    img_crop, origin_rc = crop_rect(img2d, rect_vertices)
    if img_crop.size == 0:
        print("Empty crop — ignored.")
        return

    # Create a stable crop layer name BEFORE appending lists so index stays aligned
    bead_idx_next = len(selected_bead_rois) + 1
    crop_name = f'Crop {bead_idx_next}'
    viewer.add_image(img_crop, name=crop_name)
    crop_layer_names.append(crop_name)

    # Recenter on local peak and extract fixed-size bead window
    bead_roi, peak_global_rc = recenter_and_extract_fixed(img_crop, origin_rc, bead_window=bead_window)
    if bead_roi is None:
        print("Crop too close to image border for the requested bead_window — select a larger/shifted ROI.")
        # Remove the just-added crop layer name since we didn't keep it
        try:
            viewer.layers.remove(viewer.layers[crop_name])
        except Exception:
            pass
        crop_layer_names.pop()
        return

    # Store
    selected_bead_rois.append(bead_roi)
    selected_bead_centers_global.append(peak_global_rc)

    # Update points layer (append one point)
    if len(points_layer.data) == 0:
        points_layer.data = np.array([peak_global_rc], dtype=float)
    else:
        points_layer.data = np.vstack([points_layer.data, np.array(peak_global_rc, dtype=float)])

    print(f"Added bead #{len(selected_bead_rois)} | center (r,c) = {peak_global_rc} | ROI size = {bead_roi.shape}")

@shapes_layer.events.data.connect
def on_rectangles_changed(event):
    if len(shapes_layer.data) == 0:
        return
    rect_vertices = np.array(shapes_layer.data[-1])  # last rectangle's (4,2) vertices
    handle_new_rectangle(rect_vertices)

# -------------------------------
# Delete utilities
# -------------------------------
def _delete_indices(indices_to_delete):
    """Delete beads at the given sorted indices (descending) from all registries and points layer."""
    if not indices_to_delete:
        return

    # Remove from registry lists (delete in descending order to avoid reindex issues)
    for idx in sorted(indices_to_delete, reverse=True):
        # 1) remove ROI + center
        try:
            del selected_bead_rois[idx]
            del selected_bead_centers_global[idx]
        except IndexError:
            continue

        # 2) remove crop layer if still present
        try:
            crop_name = crop_layer_names.pop(idx)
            if crop_name in [ly.name for ly in viewer.layers]:
                viewer.layers.remove(viewer.layers[crop_name])
        except Exception:
            pass

    # 3) update points layer (drop selected rows)
    if len(points_layer.data) > 0:
        mask = np.ones(len(points_layer.data), dtype=bool)
        mask[indices_to_delete] = False
        points_layer.data = points_layer.data[mask]

    print(f"Deleted {len(indices_to_delete)} bead(s). Remaining: {len(selected_bead_rois)}")

# Key: delete selected point(s)
@viewer.bind_key('D')
def delete_selected_points(viewer_):
    """Press 'D' to delete the bead(s) you selected in the 'Bead centers' layer."""
    if len(points_layer.data) == 0:
        print("No bead centers to delete.")
        return

    sel = sorted(list(points_layer.selected_data))
    if not sel:
        print("No points selected. In the 'Bead centers' layer, click a point to select it, then press 'D'.")
        return

    _delete_indices(sel)

# Key: delete the last-added bead quickly
@viewer.bind_key('Backspace')
def delete_last_bead(viewer_):
    if len(selected_bead_rois) == 0:
        print("No beads to delete.")
        return
    last_idx = len(selected_bead_rois) - 1
    _delete_indices([last_idx])


# --- Key binding: press 'A' to average all selected beads and compute PSF ---
@viewer.bind_key('A')
def average_current_beads(viewer_):
    if len(selected_bead_rois) == 0:
        print("No bead ROIs selected yet. Draw rectangles (one per bead) first.")
        return

    # Normalize & (optional) align
    rois_ready = normalize_and_align(selected_bead_rois, align_subpixel=align_subpixel,
                                     norm_mode=norm_mode, upsample_factor=80)

    # Average to PSF
    psf_avg = average_psf(rois_ready)
    viewer.add_image(psf_avg, name=f'Avg PSF (N={len(rois_ready)})', colormap='inferno')

    # FWHM
    fwhm_x_px, p_x, fwhm_y_px, p_y, prof_x, prof_y = compute_fwhm_xy_from_psf(psf_avg)
    fwhm_x_um = fwhm_x_px * pixel_size_um
    fwhm_y_um = fwhm_y_px * pixel_size_um

    print(f"\n--- Averaged PSF over {len(rois_ready)} beads ---")
    print(f"FWHM X: {fwhm_x_px:.2f} px ({fwhm_x_um:.3f} µm)")
    print(f"FWHM Y: {fwhm_y_px:.2f} px ({fwhm_y_um:.3f} µm)")

    # Plot profiles + fits
    plot_profile_and_fit(prof_x, p_x, title="Averaged PSF — X profile", xlabel="X pixel")
    plot_profile_and_fit(prof_y, p_y, title="Averaged PSF — Y profile", xlabel="Y pixel")

    # Save
    np.save("psf_average.npy", psf_avg) #saves the 2d avged psf as a numpy array in the current directory
    print("Saved averaged PSF to psf_average.npy")

napari.run()

# After closing, you still have:
print(f"Total beads selected: {len(selected_bead_rois)}")