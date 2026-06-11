#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from skimage.io import imread
from skimage import registration
import os
from datetime import datetime
from scipy.optimize import curve_fit

# -----------------------------
# USER INPUT
# -----------------------------
# 20x PSF measurement:
# image_path = '/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/260316/PSF_20x_nanobeads680-605/PSF_20x_nanobeads680-605_MMStack_Pos0_t56.tif'
# csv_path = '/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/20x_objective/PSF_crops/PSF_crops_coords.csv'
# output_dir = f"/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/20x_objective/PSF_crops/PSF_output_{datetime.now():%Y%m%d_%H%M}"
# os.makedirs(output_dir, exist_ok=True)

# # 60x PSF measurment:
# image_path = '/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/Data/260410_60x/Run03/Run03_MMStack_Pos0.ome.tif'
# csv_path = '/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/2d/60x_objective/PSF_crops/PSF_crops_coords.csv'
# output_dir = f"/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/60x_objective/PSF_crops/PSF_output_{datetime.now():%Y%m%d_%H%M}"
# os.makedirs(output_dir, exist_ok=True)

run_nr = '03'

# Upload image of beads and CSV coords
image_path = f'/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/2d/60x_objective/Run{run_nr}_MMStack_Pos0.ome.tif'
csv_path = f'/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/2d/60x_objective/PSF_crops/PSF_crops_coords.csv'
output_dir = f"/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/2d/60x_objective/PSF_output_lateral_{datetime.now():%Y%m%d_%H%M}"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
img = imread(image_path).astype(float)
df = pd.read_csv(csv_path)
save_bool = True

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def crop_from_vertices(img, vertices):
    ys = vertices[:, 0]
    xs = vertices[:, 1]
    y0, y1 = int(np.min(ys)), int(np.max(ys))
    x0, x1 = int(np.min(xs)), int(np.max(xs))
    return img[y0:y1, x0:x1]

def normalize(im):
    im = im - np.min(im)
    if np.max(im) > 0:
        im = im / np.max(im)
    return im

def extract_fixed_window(img, center_rc, window_size=21):
    """Return a window_size x window_size patch around center_rc (row, col)."""
    half = window_size // 2
    r, c = center_rc
    r1, r2 = r - half, r + half + 1
    c1, c2 = c - half, c + half + 1

    # handle edges by padding if necessary
    patch = np.zeros((window_size, window_size), dtype=img.dtype)
    H, W = img.shape
    r1_clip, r2_clip = max(0, r1), min(H, r2)
    c1_clip, c2_clip = max(0, c1), min(W, c2)
    patch[(r1_clip - r1):(r2_clip - r1), (c1_clip - c1):(c2_clip - c1)] = img[r1_clip:r2_clip, c1_clip:c2_clip]
    return patch

def align_bead_crosscorr(bead, reference):
    shift_est, _, _ = registration.phase_cross_correlation(reference, bead, upsample_factor=1)
    return shift(bead, shift=shift_est, mode='nearest', order=1)

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def normalize_profile(p):
    p = p - np.min(p)
    return p / np.max(p) if np.max(p) > 0 else p
#%%
# -----------------------------
# PARAMETERS
# -----------------------------
bead_window = 31  # size of extracted bead window
pixel_size_um = 6.5  # INPUT MANUALLY: your pixel size in um
magnification = 60  
pixel_size_true = pixel_size_um / magnification  # um

# Abbe diffraction limit
NA = 1.2
wavelength_um = 0.605  # INPUT MANUALLY: wavelength in um
rayleigh_crit_um = 0.61*wavelength_um / NA


# -----------------------------
# PROCESS BEADS
# -----------------------------
beads = []
profiles_x = []
profiles_y = []

grouped = df.groupby("index")

# First, extract all fixed-size windows around peak
for i, group in grouped:
    vertices = group[["axis-0", "axis-1"]].values
    crop = crop_from_vertices(img, vertices)
    if crop.size == 0:
        continue

    # find peak inside crop
    local_peak = np.unravel_index(np.argmax(crop), crop.shape)
    global_peak = (int(local_peak[0] + np.min(vertices[:,0])),
                   int(local_peak[1] + np.min(vertices[:,1])))

    bead_patch = extract_fixed_window(img, global_peak, window_size=bead_window)
    bead_patch = normalize(bead_patch)
    bead_patch -= np.median(bead_patch)
    bead_patch[bead_patch < 0] = 0

    beads.append(bead_patch)

# -----------------------------
# ALIGN BEADS
# -----------------------------
if len(beads) > 1:
    reference_bead = beads[0]
    beads = [align_bead_crosscorr(b, reference_bead) for b in beads]

# -----------------------------
# PROFILES
# -----------------------------
for bead in beads:
    cy, cx = bead_window // 2, bead_window // 2
    profiles_x.append(bead[cy, :])
    profiles_y.append(bead[:, cx])

# -----------------------------
# AVERAGE PSF
# -----------------------------
avg_psf = np.mean(beads, axis=0)
avg_profile_x = np.mean(profiles_x, axis=0)
avg_profile_y = np.mean(profiles_y, axis=0)
avg_profile_x_norm = normalize_profile(avg_profile_x)
avg_profile_y_norm = normalize_profile(avg_profile_y)

# -----------------------------
# SAVE OUTPUT
# -----------------------------
if save_bool:
    np.save(f"{output_dir}/avg_psf.npy", avg_psf)
    np.save(f"{output_dir}/avg_profile_x.npy", avg_profile_x)
    np.save(f"{output_dir}/avg_profile_y.npy", avg_profile_y)
    pd.DataFrame(avg_psf).to_csv(f"{output_dir}/avg_psf.csv", index=False)
#%%
# -----------------------------
# VISUALIZATION
# -----------------------------
# plt.figure(figsize=(5,5))
# for b in beads[:20]:
#     plt.imshow(b, alpha=0.2, cmap='hot')
# plt.title("Overlay of aligned beads")
# plt.show()

# plt.figure()
# plt.imshow(avg_psf, cmap='hot')
# plt.title("Average PSF")
# plt.colorbar()
# plt.savefig(f"{output_dir}/avg_psf.png")

# plt.figure()
# plt.plot(avg_profile_x, label="X")
# plt.plot(avg_profile_y, label="Y")
# plt.legend()
# plt.title("PSF Profiles")
# if save_bool:
#     plt.savefig(f"{output_dir}/avg_profiles.png")

plt.figure()
x_axis = np.arange(bead_window) * pixel_size_true
x_axis -= x_axis[bead_window // 2]  # center at 0
plt.plot(x_axis, avg_profile_x_norm, label="X")
plt.plot(x_axis, avg_profile_y_norm, label="Y")
plt.legend()
plt.title("PSF Profiles")
plt.xlabel("Distance (µm)")
plt.ylabel("Intensity")
if save_bool:
    plt.savefig(f"{output_dir}/avg_profiles.png")

print(f"Processed {len(beads)} beads")
#%%
# -----------------------------
# GAUSSIAN FIT + FWHM
# -----------------------------
# This whole thing needs to be fixed
x_axis = np.arange(bead_window) * pixel_size_true
x_axis -= x_axis[bead_window // 2]  # center at 0

# Fit Gaussians
params_x, _ = curve_fit(gaussian, x_axis, avg_profile_x_norm, p0=[1, 0, 0.5], maxfev=2000)
params_y, _ = curve_fit(gaussian, x_axis, avg_profile_y_norm, p0=[1, 0, 0.5], maxfev=2000)

fwhm_x = 2 * np.sqrt(2 * np.log(2)) * abs(params_x[2])
fwhm_y = 2 * np.sqrt(2 * np.log(2)) * abs(params_y[2])
fwhm_x_px = fwhm_x / pixel_size_true
fwhm_y_px = fwhm_y / pixel_size_true

print(f"FWHM X: {fwhm_x:.3f} um")
print(f"FWHM Y: {fwhm_y:.3f} um")

# Plot fits
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(x_axis, avg_profile_x_norm, label='Data X')
ax1.plot(x_axis, gaussian(x_axis, *params_x), '--', label=f'Gaussian fit')
ax1.set_title(rf'Lateral PSF - X profile (FWHM={fwhm_x:.2f} $\mu$m, {fwhm_x_px:.1f} px)')
ax1.set_xlabel('X (µm)')
ax1.set_ylabel('Normalized intensity')
ax1.legend()

ax2.plot(x_axis, avg_profile_y_norm, label='Data Y')
ax2.plot(x_axis, gaussian(x_axis, *params_y), '--', label=f'Gaussian fit')
ax2.set_title(rf'Lateral PSF - Y profile (FWHM={fwhm_y:.2f} $\mu$m, {fwhm_y_px:.1f} px)')
ax2.set_xlabel('Y (µm)')
ax2.set_ylabel('Normalized intensity')
ax2.legend()

plt.tight_layout()
if save_bool:
    plt.savefig(f"{output_dir}/lateral_psf_gaussian_fit.png", dpi=300, bbox_inches="tight")
plt.show()

# Save FWHM
print(save_bool)
if save_bool:
    pd.DataFrame([{
        "fwhm_x_um": fwhm_x, "fwhm_x_px": fwhm_x_px,
        "fwhm_y_um": fwhm_y, "fwhm_y_px": fwhm_y_px,
        "rayleigh_crit_um": rayleigh_crit_um
    }]).to_csv(f"{output_dir}/fwhm_lateral.csv", index=False)


# %%
