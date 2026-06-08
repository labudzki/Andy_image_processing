import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from skimage import registration
import os
from datetime import datetime
from tifffile import TiffFile

# -----------------------------
# USER INPUT
# -----------------------------
image_path = '/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/3d/260518_Run09/Run09_MMStack_Pos0.ome.tif'
csv_path = '/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/3d/260518_Run09/PSF_crops_coords.csv'
output_dir = f"/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/3d/260518_Run09/PSF_output_3d_{datetime.now():%Y%m%d_%H%M}"
os.makedirs(output_dir, exist_ok=True)

z_interval_um = 0.2   # physical z step size in microns, used for correct aspect ratio in plots
xy_window = 21        # lateral crop size in pixels (odd number so there is a well-defined center pixel)
z_window = 31         # axial crop size in z slices (odd, and large enough to capture full PSF axial extent)
save_bool = False      # set to False to skip saving outputs

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
    peak location, background subtract, clip, and normalize.
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
# SAVE
# -----------------------------
if save_bool:
    np.save(f"{output_dir}/psf_3d.npy", psf_3d)
    print(f"Saved to {output_dir}/psf_3d.npy")

# -----------------------------
# VISUALIZATION
# -----------------------------
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