# %% Imports and functions

import napari
import numpy as np
import pandas as pd
from tifffile import TiffFile
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.ndimage import shift
import matplotlib.pyplot as plt
from skimage import registration

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
# Utilities
# -------------------------------
def crop_rect(img, rect_vertices):
    """Crop an image using a napari rectangle (4x2 vertices in (row, col))."""
    y_min = int(np.floor(np.min(rect_vertices[:,0])))
    y_max = int(np.ceil (np.max(rect_vertices[:,0])))
    x_min = int(np.floor(np.min(rect_vertices[:,1])))
    x_max = int(np.ceil (np.max(rect_vertices[:,1])))
    return img[y_min:y_max, x_min:x_max], (y_min, x_min)

def recenter_and_extract_fixed(img2d, img_crop, origin_rc, bead_window=21):
    """Find peak inside crop, extract fixed-size bead window from the full image."""
    local_peak_rc = np.unravel_index(np.argmax(img_crop), img_crop.shape)
    peak_r_global = origin_rc[0] + local_peak_rc[0]
    peak_c_global = origin_rc[1] + local_peak_rc[1]

    half = bead_window // 2
    r1 = peak_r_global - half
    r2 = peak_r_global + half + 1
    c1 = peak_c_global - half
    c2 = peak_c_global + half + 1

    H, W = img2d.shape
    if r1 < 0 or c1 < 0 or r2 > H or c2 > W:
        return None, (peak_r_global, peak_c_global)

    bead_roi = img2d[r1:r2, c1:c2].astype(np.float32)
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

def plot_bead_rois_grid(rois, cols=6, cmap='inferno'):
    """
    Plot extracted bead ROIs in a grid for visual inspection.
    """
    n = len(rois)
    if n == 0:
        print("No ROIs to plot.")
        return

    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 2, rows * 2))

    for i, roi in enumerate(rois):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(roi, cmap=cmap)
        ax.set_title(f"#{i+1}")
        ax.axis('off')

    plt.suptitle(f"Extracted bead ROIs (N={n})")
    plt.tight_layout()
    plt.show()

# -------------------------------
# NEW: load rectangles from napari CSV export
# -------------------------------
def load_rectangles_from_csv(csv_path):
    """
    Read a napari shapes-layer CSV export and return a list of (4,2) vertex arrays,
    one per shape. Only rectangles are kept; other shape types are skipped.

    Expected columns: index, shape-type, vertex-index, axis-0, axis-1
    """
    df = pd.read_csv(csv_path)
    required = {'index', 'shape-type', 'axis-0', 'axis-1'}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV does not look like a napari shapes export. "
            f"Expected columns {required}, got {list(df.columns)}"
        )

    rectangles = []
    for shape_idx, group in df.groupby('index', sort=True):
        shape_type = group['shape-type'].iloc[0]
        if shape_type != 'rectangle':
            print(f"Shape {shape_idx} is '{shape_type}', skipping (only rectangles supported).")
            continue
        verts = group[['axis-0', 'axis-1']].values  # (4, 2) — (row, col)
        rectangles.append(verts)

    print(f"Loaded {len(rectangles)} rectangle(s) from {csv_path}")
    return rectangles

# %% ── PARAMETERS ──────────────────────────────────────────────────────────────
pixel_size_sensor = 6.5   # µm
magnification = 60
pixel_size_um = get_pixel_size_um(pixel_size_sensor, magnification)

lambda_um = 0.605         # µm
NA = 0.60
print(f"Abbe limit: {abbe_limit(lambda_um, NA):.3f} µm")
print(f"Rayleigh:   {rayleigh_criterion(lambda_um, NA):.3f} µm")
print(f"Sparrow:    {sparrow_limit(lambda_um, NA):.3f} µm")

bead_window     = 21      # fixed extracted bead ROI size (odd number)
align_subpixel  = True
norm_mode       = "max"   # 'max' or 'sum'

# %% ── PATHS ───────────────────────────────────────────────────────────────────
file_path = Path(r"/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/260316/PSF_20x_nanobeads680-605/PSF_20x_nanobeads680-605_MMStack_Pos0_t56.tif")
csv_path  = Path(r"/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/260316/PSF_crops/PSF_crops.csv")   

# %% ── LOAD IMAGE ──────────────────────────────────────────────────────────────
with TiffFile(file_path) as tif:
    stack = tif.asarray()
img2d = stack[0].astype('float32') if stack.ndim > 2 else stack.astype('float32')

# %% ── PROCESS RECTANGLES FROM CSV ────────────────────────────────────────────
rectangles = load_rectangles_from_csv(csv_path)

selected_bead_rois            = []
selected_bead_centers_global  = []

for i, rect_vertices in enumerate(rectangles):
    img_crop, origin_rc = crop_rect(img2d, rect_vertices)
    if img_crop.size == 0:
        print(f"Rectangle {i}: empty crop — skipped.")
        continue

    bead_roi, peak_global_rc = recenter_and_extract_fixed(
        img2d, img_crop, origin_rc, bead_window=bead_window
    )
    if bead_roi is None:
        print(f"Rectangle {i}: crop too close to image border — skipped.")
        continue

    selected_bead_rois.append(bead_roi)
    selected_bead_centers_global.append(peak_global_rc)
    print(f"Bead #{i+1} | center (r,c) = {peak_global_rc} | ROI shape = {bead_roi.shape}")

print(f"\nTotal beads loaded: {len(selected_bead_rois)}")
plot_bead_rois_grid(selected_bead_rois)
# %% ── AVERAGE & COMPUTE PSF ───────────────────────────────────────────────────
if len(selected_bead_rois) == 0:
    print("No valid bead ROIs — check your CSV path and rectangle positions.")
else:
    rois_ready = normalize_and_align(
        selected_bead_rois, align_subpixel=align_subpixel,
        norm_mode=norm_mode, upsample_factor=80
    )
    psf_avg = average_psf(rois_ready)

    fwhm_x_px, p_x, fwhm_y_px, p_y, prof_x, prof_y = compute_fwhm_xy_from_psf(psf_avg)
    fwhm_x_um = fwhm_x_px * pixel_size_um
    fwhm_y_um = fwhm_y_px * pixel_size_um

    print(f"\n--- Averaged PSF over {len(rois_ready)} beads ---")
    print(f"FWHM X: {fwhm_x_px:.2f} px ({fwhm_x_um:.3f} µm)")
    print(f"FWHM Y: {fwhm_y_px:.2f} px ({fwhm_y_um:.3f} µm)")

    plot_profile_and_fit(prof_x, p_x, title="Averaged PSF — X profile", xlabel="X pixel")
    plot_profile_and_fit(prof_y, p_y, title="Averaged PSF — Y profile", xlabel="Y pixel")

    np.save("psf_average.npy", psf_avg)
    print("Saved averaged PSF to psf_average.npy")

    # Optional: view result in napari
    viewer = napari.Viewer()
    viewer.add_image(img2d, name='Raw img', axis_labels=['Y','X'])
    viewer.add_points(
        np.array(selected_bead_centers_global, dtype=float),
        name='Bead centers', size=6, face_color='yellow'
    )
    viewer.add_image(psf_avg, name=f'Avg PSF (N={len(rois_ready)})', colormap='inferno')
    napari.run()