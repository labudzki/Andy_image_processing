
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from skimage.io import imread
from skimage import registration
import os
from datetime import datetime
from tifffile import TiffFile
import seaborn as sns

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


# # -----------------------------
# # USER INPUT
# # -----------------------------

# Upload image of beads and CSV coords
image_path = '/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/3d/260518_Run09/Run09_MMStack_Pos0.ome.tif'
csv_path = '/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/3d/260518_Run09/PSF_crops_coords.csv'
output_dir = f"/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/3d/260518_Run09/PSF_output_{datetime.now():%Y%m%d_%H%M}"
os.makedirs(output_dir, exist_ok=True)


# # -----------------------------
# # LOAD DATA
# # -----------------------------

# Read img, check shape and axes order
with TiffFile(image_path) as tif:
    stack = tif.asarray()
    # print(stack.dtype)
    print("Shape:", stack.shape)
    # print("Axes:", tif.series[0].axes)
    stack = stack.astype('float32')

# Read csv file, check columns
df = pd.read_csv(csv_path)
# print(df.columns.tolist())
# print(df.head())


# # img = imread(image_path).astype(float)
# # df = pd.read_csv(csv_path)


# -----------------------------
# PARAMETERS
# -----------------------------
bead_window = 21  # size of extracted bead window

# -----------------------------
# PROCESS BEADS
# -----------------------------
beads = [] # list of dicts to track z and bead index
profiles_x = []
profiles_y = []
profiles_z = []

grouped = df.groupby("index") # splits dataframe by the index column, where each group corresponds to one bead. We can then iterate over these groups to process each bead separately.
# for i, group in grouped:
#     print(f"Bead {i}:")
#     print(group)
#     print()
# print(grouped.ngroups)      # how many beads
# print(grouped.groups)       # dict of {bead_id: row indices}

# For each bead, find the reference z slice where the bead is brightest, find this brightest value, and its x-y location
reference_z = {}

for i, group in grouped:
    vertices = group[["axis-1", "axis-2"]].values #skip axis 0 which is z
    best_val = -np.inf
    best_z = None
    local_peak = None
    global_peak = None

    for z in range(stack.shape[0]):
        # print("here")
        img = stack[z, :, :]
        crop = crop_from_vertices(img, vertices)
        # print("crop shape:", crop.shape, "vertices:", vertices)
        if crop.size == 0:
            continue
        peak_val = np.max(crop)
        # print(i, z, peak_val)
        if peak_val > best_val:
            best_val = peak_val
            best_z = z
            local_peak = np.unravel_index(np.argmax(crop), crop.shape)
            global_peak = (int(local_peak[0] + np.min(vertices[:,0])),
                        int(local_peak[1] + np.min(vertices[:,1])))

    reference_z[i] = {
        "bead_id": i,
        "best_z": best_z,
        "peak_val": best_val,
        "best_z_peak_yx_loc_global": global_peak
    }

# -----------------------------
# CREATE AXIAL PSF PROFILES
# -----------------------------

# for each bead, take the x-y location of the peak at the reference z, and extract the intensity values across all z planes at this x-y location to create an axial profile
for i, group in grouped:
    bead_info = reference_z[i]
    peak_yx = bead_info["best_z_peak_yx_loc_global"]
    if peak_yx is None:
        print(f"Bead {i}: no valid peak found, skipping")
        continue
    z_profile = []
    for z in range(stack.shape[0]):
        img = stack[z, :, :]
        if 0 <= peak_yx[0] < img.shape[0] and 0 <= peak_yx[1] < img.shape[1]:
            z_profile.append(img[peak_yx[0], peak_yx[1]])
            # print(img[peak_yx[0], peak_yx[1]])
        else:
            z_profile.append(0)  # or np.nan if you prefer

    profiles_z.append({"bead_id": i, "profile": z_profile})
    

# Find the average axial profile across all beads
avg_profile_z = np.mean([entry["profile"] for entry in profiles_z], axis=0)


# # -----------------------------
# # SAVE OUTPUT
# # -----------------------------
np.save(f"{output_dir}/avg_axial_psf.npy", avg_profile_z)
# np.save(f"{output_dir}/avg_profile_x.npy", avg_profile_x)
# np.save(f"{output_dir}/avg_profile_y.npy", avg_profile_y)
pd.DataFrame(avg_profile_z, columns=["intensity"]).to_csv(f"{output_dir}/avg_axial_psf.csv", index=False)



# # -----------------------------
# # VISUALIZATION
# # -----------------------------

# plt.figure()
# plt.plot(avg_profile_z, label="Average Z Profile", color='black', linewidth=2)
# for entry in profiles_z:
#     plt.plot(entry["profile"], label=f"Bead {entry['bead_id']}")
#     # print(i, "here") 
# plt.legend()
# plt.title("Axial PSF Profiles")
# plt.xlabel("Z slice")
# plt.ylabel("Intensity")
# plt.show()

sns.set_theme(style="darkgrid")

fig, ax = plt.subplots()

# individual bead profiles
for entry in profiles_z:
    ax.plot(entry["profile"], label=f"Bead {entry['bead_id']}", alpha=0.6)

# average profile on top
ax.plot(avg_profile_z, label="Average Z Profile", color='black', linewidth=2)

ax.set_title("Axial PSF Profiles")
ax.set_xlabel("Z slice")
ax.set_ylabel("Intensity")
ax.legend()
plt.savefig(f"{output_dir}/avg_axial_psf_profiles.png", dpi=300)
plt.show()


# # Find XY peak location at reference z, then extract bead patch and max intensity per z at peak location
# for i, group in grouped:
#     vertices = group[["axis-0", "axis-1"]].values
#     crop = crop_from_vertices(img, vertices)
#     if crop.size == 0:
#         continue

#     # find peak inside crop
#     local_peak = np.unravel_index(np.argmax(crop), crop.shape)
#     global_peak = (int(local_peak[0] + np.min(vertices[:,0])),
#                    int(local_peak[1] + np.min(vertices[:,1])))

#     bead_patch = extract_fixed_window(img, global_peak, window_size=bead_window)
#     bead_patch = normalize(bead_patch)
#     bead_patch -= np.median(bead_patch)
#     bead_patch[bead_patch < 0] = 0

#     beads.append({
#         "patch": bead_patch,
#         "bead_id": i,
#         "z": z,
#         "peak_yx": global_peak,
#         "local_peak_yx": local_peak
#     })

# Obtaining z values
# for each bead patch
    # for each z plane
        #extract intensity value from local peak 


# # -----------------------------
# # ALIGN BEADS
# # -----------------------------
# if len(beads) > 1:
#     reference_bead = beads[0]
#     beads = [align_bead_crosscorr(b, reference_bead) for b in beads]

# # -----------------------------
# # PROFILES
# # -----------------------------
# for bead in beads:
#     cy, cx = bead_window // 2, bead_window // 2
#     profiles_x.append(bead[cy, :])
#     profiles_y.append(bead[:, cx])

# # -----------------------------
# # AVERAGE PSF
# # -----------------------------
# avg_psf = np.mean(beads, axis=0)
# avg_profile_x = np.mean(profiles_x, axis=0)
# avg_profile_y = np.mean(profiles_y, axis=0)

# # -----------------------------
# # SAVE OUTPUT
# # -----------------------------
# np.save(f"{output_dir}/avg_psf.npy", avg_psf)
# np.save(f"{output_dir}/avg_profile_x.npy", avg_profile_x)
# np.save(f"{output_dir}/avg_profile_y.npy", avg_profile_y)
# pd.DataFrame(avg_psf).to_csv(f"{output_dir}/avg_psf.csv", index=False)

# # -----------------------------
# # VISUALIZATION
# # -----------------------------
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
# plt.savefig(f"{output_dir}/avg_profiles.png")

# print(f"Processed {len(beads)} beads")

