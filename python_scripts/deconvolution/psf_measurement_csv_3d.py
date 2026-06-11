import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from scipy.optimize import curve_fit
from skimage import registration
import os
from datetime import datetime
from tifffile import TiffFile
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# USER INPUT
# -----------------------------
image_path = '/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/3d/260518_Run09/Run09_MMStack_Pos0.ome.tif'
csv_path = '/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/3d/260518_Run09/PSF_crops_coords.csv'
output_dir = f"/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/3d/260518_Run09/PSF_output_3d_{datetime.now():%Y%m%d_%H%M}"
os.makedirs(output_dir, exist_ok=True)

z_interval_um = 0.2   # physical z step size in microns, used for correct aspect ratio in plots
xy_window = 31        # lateral crop size in pixels (odd number so there is a well-defined center pixel)
z_window = 41         # axial crop size in z slices (odd, and large enough to capture full PSF axial extent)
save_bool = True      # set to False to skip saving outputs

magnfication = 60 
pixel_size_um = 6.5
pixel_size_xy_um = pixel_size_um / magnfication
print (pixel_size_xy_um)

# -----------------------------
# LOAD DATA
# -----------------------------
with TiffFile(image_path) as tif:
    stack = tif.asarray().astype('float32')  # load full 3D bead stack, shape (nz, ny, nx)
print("Stack shape:", stack.shape)

df = pd.read_csv(csv_path)  # load manually picked bead coordinates from napari/FIJI

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def crop_from_vertices(img, vertices):
    # crop a 2D bounding box from img using the min/max of the provided vertex coordinates
    ys = vertices[:, 0]
    xs = vertices[:, 1]
    y0, y1 = int(np.min(ys)), int(np.max(ys))
    x0, x1 = int(np.min(xs)), int(np.max(xs))
    return img[y0:y1, x0:x1]

def normalize(im):
    # shift to zero minimum and scale to [0, 1]
    im = im - np.min(im)
    if np.max(im) > 0:
        im = im / np.max(im)
    return im

def extract_3d_window(stack, center_zyx, z_window, xy_window):
    # cut out a (z_window, xy_window, xy_window) box centered on center_zyx
    # zero-pads if the crop extends outside the stack boundaries
    nz, ny, nx = stack.shape
    cz, cy, cx = center_zyx
    hz = z_window // 2
    hxy = xy_window // 2

    # compute crop boundaries
    z0, z1 = cz - hz, cz + hz + 1
    y0, y1 = cy - hxy, cy + hxy + 1
    x0, x1 = cx - hxy, cx + hxy + 1

    patch = np.zeros((z_window, xy_window, xy_window), dtype=stack.dtype)

    # clip boundaries to stay within the stack, then copy into patch
    z0c, z1c = max(0, z0), min(nz, z1)
    y0c, y1c = max(0, y0), min(ny, y1)
    x0c, x1c = max(0, x0), min(nx, x1)

    patch[z0c-z0 : z1c-z0,
          y0c-y0 : y1c-y0,
          x0c-x0 : x1c-x0] = stack[z0c:z1c, y0c:y1c, x0c:x1c]
    return patch

def align_3d_crosscorr(bead, reference):

    # estimate the 3D shift between bead and reference using phase cross-correlation
    # upsample_factor=10 gives subpixel precision to 1/10th of a pixel
    shift_est, _, _ = registration.phase_cross_correlation(
        reference, bead, upsample_factor=10
    )
    # apply the estimated shift to bring bead into alignment with reference
    return shift(bead, shift=shift_est, mode='nearest', order=1)

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


# -----------------------------
# MAIN FUNCTIONS
# -----------------------------

def find_bead_peaks(stack, df):
    """
    For each bead in df, scan all z-slices to find the z and (y, x)
    location of peak intensity. Returns a dict keyed by bead index.
    """
    grouped = df.groupby("index")
    reference_z = {}

    for i, group in grouped:
        vertices = group[["axis-1", "axis-2"]].values  # xy crop vertices; axis-0 is z so we skip it
        best_val = -np.inf
        best_z = None
        global_peak = None

        for z in range(stack.shape[0]):
            crop = crop_from_vertices(stack[z], vertices)  # crop the ROI at this z
            if crop.size == 0:
                continue
            peak_val = np.max(crop)
            if peak_val > best_val:  # keep track of the brightest z seen so far
                best_val = peak_val
                best_z = z
                local_peak = np.unravel_index(np.argmax(crop), crop.shape)
                # convert local crop coordinates back to global image coordinates
                global_peak = (
                    int(local_peak[0] + np.min(vertices[:, 0])),
                    int(local_peak[1] + np.min(vertices[:, 1]))
                )

        reference_z[i] = {
            "bead_id": i,
            "best_z": best_z,
            "peak_val": best_val,
            "global_peak_yx": global_peak
        }
        print(f"Bead {i}: best_z={best_z}, peak_yx={global_peak}, peak_val={best_val:.1f}")

    return reference_z


def extract_subvolumes(stack, reference_z, z_window, xy_window):
    """
    For each bead in reference_z, crop a 3D subvolume centered on the
    peak location, subtract background, clip, and normalize.
    Returns a list of dicts with bead_id and volume.
    """
    subvolumes = []

    for i, info in reference_z.items():
        if info["global_peak_yx"] is None or info["best_z"] is None:
            print(f"Bead {i}: skipping, no valid peak")
            continue

        center_zyx = (info["best_z"], info["global_peak_yx"][0], info["global_peak_yx"][1])
        vol = extract_3d_window(stack, center_zyx, z_window, xy_window)

        vol = vol - np.median(vol)  # subtract background estimated from median of the subvolume
        vol[vol < 0] = 0            # clip negatives introduced by background subtraction
        vol = normalize(vol)        # normalize to [0, 1] so bright beads don't dominate the average

        subvolumes.append({"bead_id": i, "volume": vol, "center_zyx": center_zyx})

    print(f"Extracted {len(subvolumes)} subvolumes")
    return subvolumes


def align_subvolumes(subvolumes):
    """
    Align all subvolumes to the first bead using 3D phase cross-correlation.
    This is necessary because beads don't land exactly on pixel centers —
    averaging without alignment would blur the PSF.
    """
    if len(subvolumes) < 2:
        print("Only one subvolume, skipping alignment")
        return subvolumes

    reference_vol = subvolumes[0]["volume"]  # use the first bead as the alignment target
    for entry in subvolumes:
        entry["volume"] = align_3d_crosscorr(entry["volume"], reference_vol)
    print(f"Aligned {len(subvolumes)} subvolumes to bead {subvolumes[0]['bead_id']}")

    return subvolumes


# -----------------------------
# RUN PIPELINE
# -----------------------------
reference_z = find_bead_peaks(stack, df)       # step 1: find 3D peak location for each bead
subvolumes = extract_subvolumes(stack, reference_z, z_window, xy_window)  # step 2: crop 3D subvolumes
subvolumes = align_subvolumes(subvolumes)       # step 3: align subvolumes before averaging


# -----------------------------
# AVERAGE AND NORMALIZE
# -----------------------------
psf_3d = np.mean([entry["volume"] for entry in subvolumes], axis=0)  # average aligned subvolumes
psf_3d = psf_3d / psf_3d.sum()  # normalize to sum=1, required for RL deconvolution

print("3D PSF shape:", psf_3d.shape)
# sanity check: peak should be close to the center of the array
# if it's significantly off, the alignment step didn't work correctly
print("PSF peak location:", np.unravel_index(np.argmax(psf_3d), psf_3d.shape))
print("Expected center:  ", (z_window//2, xy_window//2, xy_window//2))


# -----------------------------
# FIND AND EXTRACT FWHMs
# -----------------------------

# find PSF peak
peak_z, peak_y, peak_x = np.unravel_index(
    np.argmax(psf_3d),
    psf_3d.shape
)

# extract central profiles
profile_x = psf_3d[peak_z, peak_y, :]
profile_y = psf_3d[peak_z, :, peak_x]
profile_z = psf_3d[:, peak_y, peak_x]

# fit gaussian to each profile to find FWHM in pixels, then convert to microns
x = np.arange(xy_window) * pixel_size_xy_um
x -= x[xy_window // 2]  # center at 0

y=x

z = np.arange(z_window) * z_interval_um
z -= z[z_window // 2]

parameters_x, _ = curve_fit(gaussian, x, profile_x)
fit_x_A, fit_x_mu, fit_x_sigma = parameters_x
fit_x = gaussian(x, fit_x_A, fit_x_mu, fit_x_sigma)

fwhm_x_um = 2 * np.sqrt(2 * np.log(2)) * abs(fit_x_sigma)

parameters_y, _ = curve_fit(gaussian, y, profile_y)
fit_y_A, fit_y_mu, fit_y_sigma = parameters_y
fit_y = gaussian(y, fit_y_A, fit_y_mu, fit_y_sigma)

# Calculate FWHM from sigma (fit_x_sigma)
fwhm_y_um = 2 * np.sqrt(2 * np.log(2)) * abs(fit_y_sigma)

parameters_z, _ = curve_fit(gaussian, z, profile_z)
fit_z_A, fit_z_mu, fit_z_sigma = parameters_z
fit_z = gaussian(z, fit_z_A, fit_z_mu, fit_z_sigma)

# Calculate FWHM from sigma (fit_x_sigma)
fwhm_z_um = 2 * np.sqrt(2 * np.log(2)) * abs(fit_z_sigma)

fwhm_x_px = fwhm_x_um / pixel_size_xy_um
fwhm_y_px = fwhm_y_um / pixel_size_xy_um
fwhm_z_px = fwhm_z_um / z_interval_um

# -----------------------------
# SAVE
# -----------------------------
if save_bool:
    # Save 3D psf as npy file
    np.save(f"{output_dir}/psf_3d.npy", psf_3d)
    print(f"Saved to {output_dir}/psf_3d.npy")

    pd.DataFrame([{
            "peak_z": peak_z, "peak_y": peak_y, "peak_x": peak_x,
            "fwhm_x_um": fwhm_x_um, "fwhm_x_px": fwhm_x_px,
            "fwhm_y_um": fwhm_y_um, "fwhm_y_px": fwhm_y_px,
            "fwhm_z_um": fwhm_z_um, "fwhm_z_px": fwhm_z_px,
        }]).to_csv(f"{output_dir}/psf_fwhm.csv", index=False) 

# # -----------------------------
# # 2D VISUALIZATION
# # -----------------------------
# show three orthogonal slices through the center of the PSF
# XZ and YZ use aspect=z_interval_um to correct for anisotropic voxel size
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

peak_z = psf_3d.shape[0] // 2
axes[0].imshow(psf_3d[peak_z], cmap='hot')
axes[0].set_title("XY (at peak z)")
axes[0].axis('off')

peak_y = psf_3d.shape[1] // 2
axes[1].imshow(psf_3d[:, peak_y, :], cmap='hot', aspect=z_interval_um)
axes[1].set_title("XZ (axial)")
axes[1].axis('off')

peak_x = psf_3d.shape[2] // 2
axes[2].imshow(psf_3d[:, :, peak_x], cmap='hot', aspect=z_interval_um)
axes[2].set_title("YZ (axial)")
axes[2].axis('off')

plt.suptitle("Average 3D PSF")
plt.tight_layout()
if save_bool:
    plt.savefig(f"{output_dir}/psf_3d_projections.png", dpi=300)
plt.show()

# -----------------------------
# CENTRAL PROFILES PLOT
# -----------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(x, profile_x, lw=2)
axes[0].set_title(rf"X profile (center) FWHM = {fwhm_x_um:.3f} µm")
axes[0].set_xlabel("X (µm)")
axes[0].set_ylabel("Intensity")

axes[1].plot(y, profile_y, lw=2)
axes[1].set_title(rf"Y profile (center) FWHM = {fwhm_y_um:.3f} µm")
axes[1].set_xlabel("Y (µm)")
axes[1].set_ylabel("Intensity")

axes[2].plot(z, profile_z, lw=2)
axes[2].set_title(rf"Z profile (center) FWHM = {fwhm_z_um:.3f} µm")
axes[2].set_xlabel("Z (µm)")
axes[2].set_ylabel("Intensity")

plt.suptitle("Central PSF Profiles")
plt.tight_layout()

if save_bool:
    plt.savefig(f"{output_dir}/psf_central_profiles.png", dpi=300)

plt.show()

# -----------------------------
# 3D VISUALIZATION
# -----------------------------

# use a threshold to only show voxels above a fraction of the peak
# adjust this value to show more or less of the PSF
# threshold = 0.2

# # get coordinates and intensities of voxels above threshold
# z_coords, y_coords, x_coords = np.where(psf_3d > threshold * psf_3d.max())
# intensities = psf_3d[z_coords, y_coords, x_coords]

# # convert z coordinates to microns for correct aspect ratio
# z_coords_um = z_coords * z_interval_um

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')

# print("here")

# sc = ax.scatter(
#     x_coords, y_coords, z_coords_um,
#     c=intensities,
#     cmap='hot',
#     alpha=0.3,          # transparency so internal structure is visible
#     s=10,               # marker size
#     vmin=threshold * psf_3d.max(),
#     vmax=psf_3d.max()
# )

# plt.colorbar(sc, ax=ax, label="Intensity", shrink=0.5)
# ax.set_xlabel("X (px)")
# ax.set_ylabel("Y (px)")
# ax.set_zlabel("Z (µm)")
# ax.set_title("3D PSF")

# plt.tight_layout()
# if save_bool:
#     plt.savefig(f"{output_dir}/psf_3d_scatter.png", dpi=300)
# plt.show()

#Surface plot of MIP along z-axis
Z = np.max(psf_3d, axis=0)

Y, X = np.meshgrid(
    np.arange(Z.shape[0]),
    np.arange(Z.shape[1]),
    indexing="ij"
)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    X, Y, Z,
    cmap='hot',
    linewidth=0
)

fig.colorbar(surf, ax=ax, label="Max Intensity")

ax.set_xlabel("X (px)")
ax.set_ylabel("Y (px)")
ax.set_zlabel("Intensity")
ax.set_title("Maximum Intensity Projection")

plt.show()