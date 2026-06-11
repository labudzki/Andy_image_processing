
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from scipy.optimize import curve_fit
from skimage.io import imread
from skimage import registration
import os
from datetime import datetime
from tifffile import TiffFile
import seaborn as sns
from scipy.optimize import curve_fit   

#%%
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

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

# # -----------------------------
# # USER INPUT
# # -----------------------------
run_nr = '09'
save_bool = True #True if you want to save PSF info and plot

# Upload image of beads and CSV coords
image_path = f'/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/3d/260518_Run{run_nr}/Run{run_nr}_MMStack_Pos0.ome.tif'
csv_path = f'/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/3d/260518_Run{run_nr}/PSF_crops_coords.csv'
if save_bool:
    output_dir = f"/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/3d/260518_Run{run_nr}/PSF_output_axial_{datetime.now():%Y%m%d_%H%M}"
    os.makedirs(output_dir, exist_ok=True)

z_interval_um = 0.2 #um INPUT MANUALLY


wavelength_um = 0.605 
eta = 1.33 # RI of water 
NA = 1.2 # numerical aperture of the objective 
axial_resolution_limit = 2* wavelength_um * eta /(NA**2) #equation from p4 of handbook of biological confocal microscopy
print(f"Axial resolution limit: {axial_resolution_limit:.2f} um")


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
# PROCESS BEADS
# -----------------------------
beads = [] # list of dicts to track z and bead index
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

#Normalize bead profiles and average profile to max of 1 for easier comparison


normalized_profiles_z = [
    {**entry, "profile": normalize(entry["profile"])}
    for entry in profiles_z
]

avg_profile_z_normalized = normalize(avg_profile_z)
avg_profile_z_after_norm = np.mean([entry["profile"] for entry in normalized_profiles_z], axis=0)

#%% 
# --------------------
# CENTER THE PSF
# --------------------
peak_idx = np.argmax(avg_profile_z_normalized)
center_idx = len(avg_profile_z_normalized) // 2
shift_amount = center_idx - peak_idx

# Pad with 0's so no data is lost
pad = abs(shift_amount)
avg_profile_z_normalized_shifted = shift(
    np.pad(avg_profile_z_normalized, pad, mode='constant', constant_values=0),
    shift=shift_amount, mode='constant', cval=0
)
avg_profile_z_after_norm_shifted = shift(
    np.pad(avg_profile_z_after_norm, pad, mode='constant', constant_values=0),
    shift=shift_amount, mode='constant', cval=0
)

# Update z_axis to match new length
z_axis_shifted = np.arange(len(avg_profile_z_normalized_shifted)) * z_interval_um
z_axis_shifted -= z_axis_shifted[len(z_axis_shifted) // 2]  # center at 0

#%%
# # -----------------------------
# # SAVE OUTPUT
# # -----------------------------
if save_bool:
    np.save(f"{output_dir}/avg_axial_psf.npy", avg_profile_z)
    # np.save(f"{output_dir}/avg_profile_x.npy", avg_profile_x)
    # np.save(f"{output_dir}/avg_profile_y.npy", avg_profile_y)
    pd.DataFrame(avg_profile_z, columns=["intensity"]).to_csv(f"{output_dir}/avg_axial_psf.csv", index=False)

    # Shifted profile
    np.save(f"{output_dir}/avg_axial_psf_shifted.npy", avg_profile_z_normalized_shifted)
    pd.DataFrame({"z_um": z_axis_shifted, "intensity": avg_profile_z_normalized_shifted}).to_csv(f"{output_dir}/avg_axial_psf_shifted.csv", index=False)



# # -----------------------------
# # VISUALIZATION
# #  Plot individual bead profiles and average profile, colored by x and y position of the bead in the FOV
# #  Plot individual normalized bead profiles and average profile, colored by x and y position of the bead in the FOV
# #  Plot averaged bead profile with and without normalization
# # -----------------------------

# Plot without normalization

# get xy positions for color mapping
peak_positions = [(reference_z[entry["bead_id"]]["best_z_peak_yx_loc_global"]) for entry in profiles_z]
x_positions = np.array([p[1] for p in peak_positions])  # x (col)
y_positions = np.array([p[0] for p in peak_positions])  # y (row)
norm_x = plt.Normalize(vmin=x_positions.min(), vmax=x_positions.max()) #normalizes the x and y positions to match them to a colormap 
norm_y = plt.Normalize(vmin=y_positions.min(), vmax=y_positions.max())
cmap_x = plt.cm.viridis
 #plt.cm.plasma
cmap_y = cmap_x.reversed()
z_axis = np.arange(stack.shape[0]) * z_interval_um #to scale the z axis to real units (um) instead of slice index



# Compare PSF profiles colored by x position, and bead positions in the FOV colored by x position. 
sns.set_theme(style="darkgrid")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# --- Top left: axial profiles colored by x position ---
for entry, x_pos in zip(profiles_z, x_positions): # plot each bead's axial profile with color determined by x position
    ax1.plot(z_axis, entry["profile"], color=cmap_x(norm_x(x_pos)), alpha=0.7)
ax1.plot(z_axis, avg_profile_z, color='black', linewidth=2, label="Average")
sm_x = plt.cm.ScalarMappable(cmap=cmap_x, norm=norm_x) #scalarmappable object to create colorbar
sm_x.set_array([])
plt.colorbar(sm_x, ax=ax1, label="X position (px)")
ax1.set_title("Axial PSF Profiles (colored by X position)")
ax1.set_xlabel("Z (\u03bcm)")
ax1.set_ylabel("Intensity")
ax1.legend()

# --- Top right: bead positions colored by x ---
img_h, img_w = stack.shape[1], stack.shape[2]
ax2.set_xlim(0, img_w)
ax2.set_ylim(img_h, 0) # flipped so y=0 is at top, matching image coords
ax2.set_facecolor("black")
ax2.set_aspect("equal")
for entry, x_pos, y_pos in zip(profiles_z, x_positions, y_positions): #plots each bad as a dot in its (x,y) pos in the FOV. colored the same way as in ax1
    ax2.scatter(x_pos, y_pos, color=cmap_x(norm_x(x_pos)), s=80, zorder=5)
    ax2.text(x_pos + 15, y_pos, str(entry["bead_id"]), color='white', fontsize=8)
plt.colorbar(sm_x, ax=ax2, label="X position (px)")
ax2.set_title("Bead positions in FOV (colored by X position)")
ax2.set_xlabel("X (px)")
ax2.set_ylabel("Y (px)")

# --- Bottom left: axial profiles colored by y position ---
for entry, y_pos in zip(profiles_z, y_positions):
    ax3.plot(z_axis, entry["profile"], color=cmap_y(norm_y(y_pos)), alpha=0.7)
ax3.plot(z_axis, avg_profile_z, color='black', linewidth=2, label="Average")
sm_y = plt.cm.ScalarMappable(cmap=cmap_y, norm=norm_y)
sm_y.set_array([])
plt.colorbar(sm_y, ax=ax3, label="Y position (px)")
ax3.set_title("Axial PSF Profiles (colored by Y position)")
ax3.set_xlabel("Z (\u03bcm)")
ax3.set_ylabel("Intensity")
ax3.legend()

# --- Bottom right: bead positions colored by y ---
ax4.set_xlim(0, img_w)
ax4.set_ylim(img_h, 0)
ax4.set_facecolor("black")
ax4.set_aspect("equal")
for entry, x_pos, y_pos in zip(profiles_z, x_positions, y_positions):
    ax4.scatter(x_pos, y_pos, color=cmap_y(norm_y(y_pos)), s=80, zorder=5)
    ax4.text(x_pos + 15, y_pos, str(entry["bead_id"]), color='white', fontsize=8)
plt.colorbar(sm_y, ax=ax4, label="Y position (px)")
ax4.set_title("Bead positions in FOV (colored by Y position)")
ax4.set_xlabel("X (px)")
ax4.set_ylabel("Y (px)")

plt.tight_layout()
if save_bool:
    plt.savefig(f"{output_dir}/avg_axial_psf_profiles.png", dpi=300, bbox_inches="tight")
plt.show()

# --------------------
# Plot with normalization


sns.set_theme(style="darkgrid")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# --- Top left: normalized axial profiles colored by x position ---
for entry, x_pos, norm_entry in zip(profiles_z, x_positions, normalized_profiles_z):
    ax1.plot(z_axis, norm_entry["profile"],
             color=cmap_x(norm_x(x_pos)), alpha=0.7)

ax1.plot(z_axis, avg_profile_z_after_norm,
         color='black', linewidth=2, label="Average (normalized)")

sm_x = plt.cm.ScalarMappable(cmap=cmap_x, norm=norm_x)
sm_x.set_array([])
plt.colorbar(sm_x, ax=ax1, label="X position (px)")

ax1.set_title("Normalized Axial PSF Profiles (colored by X position)")
ax1.set_xlabel("Z (µm)")
ax1.set_ylabel("Normalized intensity")
ax1.legend()


# --- Top right: bead positions colored by x (same data) ---
img_h, img_w = stack.shape[1], stack.shape[2]
ax2.set_xlim(0, img_w)
ax2.set_ylim(img_h, 0)
ax2.set_facecolor("black")
ax2.set_aspect("equal")

for entry, x_pos, y_pos in zip(profiles_z, x_positions, y_positions):
    ax2.scatter(x_pos, y_pos, color=cmap_x(norm_x(x_pos)), s=80, zorder=5)
    ax2.text(x_pos + 15, y_pos, str(entry["bead_id"]), color='white', fontsize=8)

plt.colorbar(sm_x, ax=ax2, label="X position (px)")
ax2.set_title("Bead positions in FOV (colored by X position)")
ax2.set_xlabel("X (px)")
ax2.set_ylabel("Y (px)")


# --- Bottom left: normalized axial profiles colored by y position ---
for entry, y_pos, norm_entry in zip(profiles_z, y_positions, normalized_profiles_z):
    ax3.plot(z_axis, norm_entry["profile"],
             color=cmap_y(norm_y(y_pos)), alpha=0.7)

ax3.plot(z_axis, avg_profile_z_after_norm,
         color='black', linewidth=2, label="Average (normalized)")

sm_y = plt.cm.ScalarMappable(cmap=cmap_y, norm=norm_y)
sm_y.set_array([])
plt.colorbar(sm_y, ax=ax3, label="Y position (px)")

ax3.set_title("Normalized Axial PSF Profiles (colored by Y position)")
ax3.set_xlabel("Z (µm)")
ax3.set_ylabel("Normalized intensity")
ax3.legend()


# --- Bottom right: bead positions colored by y (same data) ---
ax4.set_xlim(0, img_w)
ax4.set_ylim(img_h, 0)
ax4.set_facecolor("black")
ax4.set_aspect("equal")

for entry, x_pos, y_pos in zip(profiles_z, x_positions, y_positions):
    ax4.scatter(x_pos, y_pos, color=cmap_y(norm_y(y_pos)), s=80, zorder=5)
    ax4.text(x_pos + 15, y_pos, str(entry["bead_id"]), color='white', fontsize=8)

plt.colorbar(sm_y, ax=ax4, label="Y position (px)")
ax4.set_title("Bead positions in FOV (colored by Y position)")
ax4.set_xlabel("X (px)")
ax4.set_ylabel("Y (px)")

plt.tight_layout()
if save_bool:
    plt.savefig(f"{output_dir}/avg_axial_psf_profiles_normalized.png",
                dpi=300, bbox_inches="tight")
plt.show()


sse = np.sum((avg_profile_z_normalized - avg_profile_z_after_norm) ** 2)
# print(f"Sum of squared errors between avg_profile_z_normalized and avg_profile_z_after_norm: {sse:.6f}")

# calculate root mean squared error (RMSE) between avg_profile_z_normalized and avg_profile_z_after_norm
rmse = np.sqrt(sse / len(avg_profile_z_normalized))
print(f"Root mean squared error between avg_profile_z_normalized and avg_profile_z_after_norm: {rmse:.6f}")

# Plot avg_profile_z_normalized and avg_profile_z_after_norm together to check they are the same
plt.figure()
plt.plot(z_axis_shifted, avg_profile_z_normalized_shifted, color='red', linewidth=2, label="Average Z Profile (normalized)")
plt.plot(z_axis_shifted, avg_profile_z_after_norm_shifted, color='blue', linewidth=2, label="Average of normalized profiles")
plt.legend()
plt.title(f"Average profile after normalization; center shifted, rmse = {rmse:.6f}")
# plt.title(f"Check average profile after normalization; sse = {sse:.6f}, rmse = {rmse:.6f}")
plt.xlabel("Z (µm))")
plt.ylabel("Intensity")
if save_bool:
    plt.savefig(f"{output_dir}/avg_z_profiles_normalized_and_not_normalized.png",
                dpi=300, bbox_inches="tight")
plt.show()



#%%
# --------------------
# GAUSSIAN FIT + FWHM   
# --------------------
#Fit a gaussian to the average axial profile and print the FWHM in z. 
parameters_avg, _ = curve_fit(gaussian, z_axis, avg_profile_z_normalized)
fit_A, fit_mu, fit_sigma = parameters_avg
fit_y = gaussian(z_axis, fit_A, fit_mu, fit_sigma)

# Calculate FWHM from sigma (fit_sigma)
fwhm_avg = 2 * np.sqrt(2 * np.log(2)) * abs(fit_sigma)

plt.plot(z_axis, avg_profile_z_normalized, label='Data')
plt.plot(z_axis, fit_y, '-', label='Gaussian fit')
plt.title(f'Gaussian Fit, FWHM = {fwhm_avg:.2f} um')
plt.legend()
plt.show()

if save_bool:
    fwhm_avg_px = fwhm_avg / z_interval_um
    pd.DataFrame([{"fwhm_avg_um": fwhm_avg, "fwhm_avg_px": fwhm_avg_px, "axial_resolution_limit_um": axial_resolution_limit}]).to_csv(f"{output_dir}/fwhm_avg.csv", index=False)

#%%
#Fit a gaussian to the normalizedaverage axial profile and print the FWHM in z. 
parameters_norm, _ = curve_fit(gaussian, z_axis, avg_profile_z_after_norm)
fit_A, fit_mu, fit_sigma = parameters_norm
fit_y = gaussian(z_axis, fit_A, fit_mu, fit_sigma)

# Calculate FWHM from sigma (fit_sigma)
fwhm_norm = 2 * np.sqrt(2 * np.log(2)) * abs(fit_sigma)

plt.plot(z_axis, avg_profile_z_after_norm, label='Data')
plt.plot(z_axis, fit_y, '-', label='Gaussian fit')
plt.title(f'Gaussian Fit, FWHM = {fwhm_norm:.2f} um')
plt.legend()
plt.show()



