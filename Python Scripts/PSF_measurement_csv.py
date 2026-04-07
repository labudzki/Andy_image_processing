
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from skimage.io import imread
from skimage import registration
import os
from datetime import datetime

# -----------------------------
# USER INPUT
# -----------------------------
image_path = '/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/260316/PSF_20x_nanobeads680-605/PSF_20x_nanobeads680-605_MMStack_Pos0_t56.tif'
csv_path = '/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/260316/PSF_crops/PSF_crops2_coords.csv'
output_dir = f"/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/260316/PSF_crops/PSF_output_{datetime.now():%Y%m%d_%H%M}"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
img = imread(image_path).astype(float)
df = pd.read_csv(csv_path)

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
    shift_est, _, _ = registration.phase_cross_correlation(reference, bead, upsample_factor=50)
    return shift(bead, shift=shift_est, mode='nearest', order=1)

# -----------------------------
# PARAMETERS
# -----------------------------
bead_window = 21  # size of extracted bead window

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

# -----------------------------
# SAVE OUTPUT
# -----------------------------
np.save(f"{output_dir}/avg_psf.npy", avg_psf)
np.save(f"{output_dir}/avg_profile_x.npy", avg_profile_x)
np.save(f"{output_dir}/avg_profile_y.npy", avg_profile_y)
pd.DataFrame(avg_psf).to_csv(f"{output_dir}/avg_psf.csv", index=False)

# -----------------------------
# VISUALIZATION
# -----------------------------
plt.figure(figsize=(5,5))
for b in beads[:20]:
    plt.imshow(b, alpha=0.2, cmap='hot')
plt.title("Overlay of aligned beads")
plt.show()

plt.figure()
plt.imshow(avg_psf, cmap='hot')
plt.title("Average PSF")
plt.colorbar()
plt.savefig(f"{output_dir}/avg_psf.png")

plt.figure()
plt.plot(avg_profile_x, label="X")
plt.plot(avg_profile_y, label="Y")
plt.legend()
plt.title("PSF Profiles")
plt.savefig(f"{output_dir}/avg_profiles.png")

print(f"Processed {len(beads)} beads")

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.ndimage import shift
# from skimage.io import imread
# import os
# from datetime import datetime
# from skimage import registration
# from scipy.ndimage import shift


# # -----------------------------
# # USER INPUT
# # -----------------------------
# image_path = '/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/260316/PSF_20x_nanobeads680-605/PSF_20x_nanobeads680-605_MMStack_Pos0_t56.tif'
# csv_path = '/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/260316/PSF_crops/PSF_crops2_coords.csv'
# output_dir = f"/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/260316/PSF_crops/PSF_output_{datetime.now():%Y%m%d_%H%M}"
# os.makedirs(output_dir, exist_ok=True)

# # -----------------------------
# # LOAD DATA
# # -----------------------------
# img = imread(image_path).astype(float)
# df = pd.read_csv(csv_path)

# # -----------------------------
# # HELPER FUNCTIONS
# # -----------------------------
# def crop_from_vertices(img, vertices):
#     """
#     vertices: Nx2 array [[y,x], ...] or [[axis-0, axis-1]]
#     """
#     ys = vertices[:, 0]
#     xs = vertices[:, 1]

#     y0, y1 = int(np.min(ys)), int(np.max(ys))
#     x0, x1 = int(np.min(xs)), int(np.max(xs))

#     return img[y0:y1, x0:x1]

# def center_of_mass(im):
#     y, x = np.indices(im.shape)
#     total = im.sum()
#     return (x * im).sum() / total, (y * im).sum() / total

# # def align_bead(im):
# #     cx, cy = center_of_mass(im)
# #     shift_x = im.shape[1] // 2 - cx
# #     shift_y = im.shape[0] // 2 - cy
# #     return shift(im, shift=(shift_y, shift_x), mode='nearest')

# def align_bead_peak(im):
#     peak_y, peak_x = np.unravel_index(np.argmax(im), im.shape)
#     center_y = im.shape[0] // 2
#     center_x = im.shape[1] // 2
#     shift_y = center_y - peak_y
#     shift_x = center_x - peak_x
#     return shift(im, shift=(shift_y, shift_x), order=3, mode='nearest')

# # def align_bead_subpixel(im):
# #     # Threshold to isolate peak region
# #     thresh = im > (0.5 * im.max())

# #     if np.sum(thresh) < 5:
# #         # fallback if threshold too small
# #         cy, cx = np.unravel_index(np.argmax(im), im.shape)
# #     else:
# #         cy, cx = center_of_mass(im * thresh)

# #     center_y = im.shape[0] / 2
# #     center_x = im.shape[1] / 2

# #     shift_y = center_y - cy
# #     shift_x = center_x - cx

# #     return shift(im, shift=(shift_y, shift_x), order=3, mode='nearest')


# def align_bead_crosscorr(bead, reference):
#     shift_est, _, _ = registration.phase_cross_correlation(reference, bead, upsample_factor=50)
#     return shift(bead, shift=shift_est, mode='nearest', order=1)

# def normalize(im):
#     im = im - np.min(im)
#     if np.max(im) > 0:
#         im = im / np.max(im)
#     return im

# # -----------------------------
# # PROCESS BEADS
# # -----------------------------
# beads = []
# profiles_x = []
# profiles_y = []

# grouped = df.groupby("index")

# for i, group in grouped:
#     # Extract vertices
#     vertices = group[["axis-0", "axis-1"]].values

#     bead = crop_from_vertices(img, vertices)

#     # Skip edge cases
#     if bead.size == 0:
#         continue
#     # enforce odd size
#     if bead.shape[0] % 2 == 0:
#         bead = bead[:-1, :]
#     if bead.shape[1] % 2 == 0:
#         bead = bead[:, :-1]

#     bead = normalize(bead)
#     # bead = align_bead(bead)
#     bead = bead - np.median(bead)
#     bead[bead < 0] = 0

#     beads.append(bead)

#     # pick the first bead as reference
#     reference_bead = beads[0]

#     # align each bead to reference
#     aligned_beads = [align_bead_crosscorr(b, reference_bead) for b in beads]

#     # Profiles through center
#     cy, cx = bead.shape[0] // 2, bead.shape[1] // 2
#     profiles_x.append(bead[cy, :])
#     profiles_y.append(bead[:, cx])

#     # Debug image
#     plt.imsave(f"{output_dir}/bead_{i}.png", bead, cmap='hot')

# # -----------------------------
# # ENSURE SAME SIZE (important)
# # -----------------------------
# # Crop all beads to smallest common shape
# min_y = min(b.shape[0] for b in beads)
# min_x = min(b.shape[1] for b in beads)

# beads = np.array([b[:min_y, :min_x] for b in beads])
# profiles_x = np.array([p[:min_x] for p in profiles_x])
# profiles_y = np.array([p[:min_y] for p in profiles_y])

# # -----------------------------
# # VISUALIZE
# # -----------------------------
# plt.figure(figsize=(5,5))
# for i in range(min(20, len(beads))):
#     plt.imshow(beads[i], alpha=0.2, cmap='hot')
# plt.title("Overlay of aligned beads")
# plt.show()

# # -----------------------------
# # AVERAGE PSF
# # -----------------------------
# avg_psf = np.mean(beads, axis=0)
# avg_profile_x = profiles_x.mean(axis=0)
# avg_profile_y = profiles_y.mean(axis=0)

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