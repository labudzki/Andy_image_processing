"""
Richardson-Lucy / Wiener Deconvolution of 3D Confocal AMF Stacks
------------------------------------------------------------------
Loads a 3D confocal image stack (OME-TIFF) and an experimentally measured
PSF, matches the PSF's z-sampling to the data via binning (energy-
conserving), and applies either Richardson-Lucy or Wiener deconvolution
at a configurable set of parameter values for comparison.

Workflow:
1. Load raw image stack and normalize.
2. Load experimental PSF, crop in z, and resample (bin) to match the
   data's z-interval.
3. Deconvolve using the method set by `method` ('RL' or 'wiener'), looping
   over `n_iter_list` (RL) or `balance_list` (Wiener).
4. Visualize original vs. deconvolved volumes in 2D (xy/xz/yz planes)
   and in 3D (napari), for however many parameter values are specified.

To change method or parameter values, edit `method`, `n_iter_list`, or
`balance_list` — stats, 2D plots, and the napari viewer update
automatically.
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imwrite, TiffFile
from pathlib import Path
from scipy.signal import convolve2d as conv2
from scipy.ndimage import zoom
from skimage import color, data, restoration
from PIL import Image
from datetime import datetime
import napari


# ---------------------
# Import 60X 3D data
# ---------------------
path_movie = Path(
    '/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/RawData/260702/CFL2605A123/Run04_150430/Stack_Run04_150430/Stack_Run04_150430_MMStack_Pos0.ome.tif'
)

with TiffFile(path_movie) as tif:
    stack = tif.asarray()
# print(stack.shape)

#%%
psf_type = "exp"
# psf_type = "num"
z_int_data = 0.5  # um
slice_index_t = 8
stack_single_timeframe = stack[:28, :, :]

# print(stack_single_timeframe.shape)

my_stack = stack_single_timeframe

# # save my_stack
# output_path_stack = '/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/Analysis/260702/CFL2605A123/Run04'
# imwrite(output_path_stack + "/Run04_singleframe.tif", my_stack)


# Normalize
my_stack_norm = my_stack / my_stack.max()

#%%

# ---------------------
# Import experimental PSF
# ---------------------

psf = np.load('/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/PSF/Nikon_CFI_PlanApo_VC_60X_WI/psf_3d.npy')
z_int_psf = 0.1  # um
output_path_psf = '/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/PSF/Nikon_CFI_PlanApo_VC_60X_WI'
# imwrite(output_path_psf + "/psf_3d.tif", psf)

print("\n original PSF")
print(f"PSF shape: {psf.shape}")
print(f"PSF min:  {psf.min():.4g}")
print(f"PSF max:  {psf.max():.4g}")
print(f"PSF sum:  {psf.sum():.4g}")
print(f"PSF mean: {psf.mean():.4g}\n")

# Crop the PSF in z so the kernel is smaller than the image
id_mid = psf.shape[0] // 2
my_range = int(12 / (0.1 * 2))
# print(id_mid, my_range)
psf_cropped = psf[id_mid - my_range:id_mid + my_range, :, :]


print("\n cropped PSF ")
print(f"PSF shape: {psf_cropped.shape}")
print(f"PSF min:  {psf_cropped.min():.4g}")
print(f"PSF max:  {psf_cropped.max():.4g}")
print(f"PSF sum:  {psf_cropped.sum():.4g}")
print(f"PSF mean: {psf_cropped.mean():.4g}\n")

# Bin the resampling (conserves energy, unlike zoom)
factor = int(z_int_data / z_int_psf)
n_z = (psf_cropped.shape[0] // factor) * factor
psf_trimmed = psf_cropped[:n_z]
psf_resampled = psf_trimmed.reshape(
    n_z // factor,
    factor,
    *psf_trimmed.shape[1:]
).sum(axis=1)
psf_resampled_norm = psf_resampled / psf_resampled.sum()

print("\n resampled PSF ")
print(f"PSF shape: {psf_resampled.shape}")
print(f"PSF min:  {psf_resampled.min():.4g}")
print(f"PSF max:  {psf_resampled.max():.4g}")
print(f"PSF sum:  {psf_resampled.sum():.4g}")
print(f"PSF mean: {psf_resampled.mean():.4g}\n")

print("\n resampled normalized PSF ")
print(f"PSF shape: {psf_resampled_norm.shape}")
print(f"PSF min:  {psf_resampled_norm.min():.4g}")
print(f"PSF max:  {psf_resampled_norm.max():.4g}")
print(f"PSF sum:  {psf_resampled_norm.sum():.4g}")
print(f"PSF mean: {psf_resampled_norm.mean():.4g}\n")

# imwrite(output_path_psf + "/psf_3d_resampled_norm.tif", psf)

# %% uploading numerical PSF
psf_num = imread('/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/PSF/Nikon_CFI_PlanApo_VC_60X_WI/numerical_psf.tif')
z_int_psf_num = 0.5


#%%

# print("image slice shape:", my_stack_norm.shape)
# print("PSF shape:", psf.shape)
# print("PSF resampled shape:", psf_resampled_norm.shape)

psf_3d = psf_resampled_norm
# psf_3d = psf_num

peak_z = np.unravel_index(np.argmax(psf_3d), psf_3d.shape)[0]
# print(peak_z, psf_3d.shape[0] // 2)

#%%
# -------------
# Deconvolution settings
# -------------
method = 'RL'   # 'RL' for Richardson-Lucy, 'wiener' for Wiener deconvolution
n_it = 50
n_iter_list   = [n_it]           # used when method == 'RL'
balance_list  = [0.01, 0.1]        # used when method == 'wiener' (regularization strength)

output_path_data = '/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/PSF/PSF_data/deconvolution_test_data/PSF_exp'

deconvolved = {}

if method == 'RL':
    for n in n_iter_list:
        deconvolved[n] = restoration.richardson_lucy(my_stack_norm, psf_3d, num_iter=n, clip=True)
    param_list = n_iter_list
    param_label = 'iterations'
    imwrite(output_path_data + f"/deconvolved_RL{n}_clip.tif", deconvolved[n])
    print("deconvolved data saved")

elif method == 'wiener':
    for b in balance_list:
        deconvolved[b] = restoration.wiener(my_stack_norm, psf_3d, balance=b)
    param_list = balance_list
    param_label = 'balance'
    imwrite(output_path_data + f"/deconvolved_wiener{b}.tif", deconvolved[b])
    print("deconvolved data saved")

else:
    raise ValueError(f"Unknown method '{method}'. Use 'RL' or 'wiener'.")
#%%
# save deconvolved data



# checking min and max for deconvolved images
for p, img in deconvolved.items():
    print(f"{method} {p}: min={img.min():.4f}, max={img.max():.4f}")

# Stats
print(
    f"{'original':10s} "
    f"min={my_stack_norm.min():.6g}, max={my_stack_norm.max():.6g}, "
    f"mean={my_stack_norm.mean():.6g}, 99%={np.percentile(my_stack_norm, 99):.6g}, "
    f"99.9%={np.percentile(my_stack_norm, 99.9):.6g}, 99.99%={np.percentile(my_stack_norm, 99.99):.6g}"
)
for p, img in deconvolved.items():
    print(
        f"{method} {p:<7} "
        f"min={img.min():.6g}, max={img.max():.6g}, "
        f"mean={img.mean():.6g}, 99%={np.percentile(img, 99):.6g}, "
        f"99.9%={np.percentile(img, 99.9):.6g}, 99.99%={np.percentile(img, 99.99):.6g}"
    )

# print("input max:", my_stack_norm.max())
# for p, img in deconvolved.items():
#     print(f"{method} {p} max:", img.max())
#     print(f"{method} {p} > 1:", np.sum(img > 1))

#%%
# -------------
# plotting - 2D (xy plane)
# -------------
z_plane_to_show = 10
n_rows = 1 + len(param_list)

fig, ax = plt.subplots(nrows=n_rows, ncols=1, figsize=(16, 5 * n_rows / 3))
plt.gray()
for a in ax:
    a.axis('off')

ax[0].imshow(my_stack_norm[z_plane_to_show, :, :], vmin=my_stack_norm.min(), vmax=my_stack_norm.max())
ax[0].set_title('Original Data (xy plane)')

for i, p in enumerate(param_list):
    ax[i + 1].imshow(deconvolved[p][z_plane_to_show, :, :], vmin=deconvolved[p].min(), vmax=deconvolved[p].max())
    ax[i + 1].set_title(rf'{method} – {p} {param_label} (xy plane)')

fig.subplots_adjust(wspace=0.02, hspace=0.2, top=0.9, bottom=0.05, left=0, right=1)
plt.show()

#%%
# -------------
# plotting - xz plane
# -------------
y_plane_to_show = my_stack_norm.shape[1] // 2

fig, ax = plt.subplots(nrows=n_rows, ncols=1, figsize=(16, 5 * n_rows / 3))
plt.gray()
for a in ax:
    a.axis('off')

ax[0].imshow(my_stack_norm[:, y_plane_to_show, :], vmin=my_stack_norm.min(), vmax=my_stack_norm.max())
ax[0].set_title('Original Data (xz plane)')

for i, p in enumerate(param_list):
    ax[i + 1].imshow(deconvolved[p][:, y_plane_to_show, :], vmin=deconvolved[p].min(), vmax=deconvolved[p].max())
    ax[i + 1].set_title(rf'{method} – {p} {param_label} (xz plane)')

fig.subplots_adjust(wspace=0.02, hspace=0.2, top=0.9, bottom=0.05, left=0, right=1)
plt.show()

#%%
# -------------
# plotting - yz plane
# -------------
x_plane_to_show = my_stack_norm.shape[2] // 2

fig, ax = plt.subplots(nrows=n_rows, ncols=1, figsize=(16, 5 * n_rows / 3))
plt.gray()
for a in ax:
    a.axis('off')

ax[0].imshow(my_stack_norm[:, :, x_plane_to_show], vmin=my_stack_norm.min(), vmax=my_stack_norm.max())
ax[0].set_title('Original Data (yz plane)')

for i, p in enumerate(param_list):
    ax[i + 1].imshow(deconvolved[p][:, :, x_plane_to_show], vmin=deconvolved[p].min(), vmax=deconvolved[p].max())
    ax[i + 1].set_title(rf'{method} – {p} {param_label} (yz plane)')

fig.subplots_adjust(wspace=0.02, hspace=0.2, top=0.9, bottom=0.05, left=0, right=1)
plt.show()

#%%
# -------------
# plotting - 3D (napari)
# -------------
pixel_size = 6.5  # um
magnification = 60
pixel_size_true = pixel_size / magnification
my_scale = (z_int_data, pixel_size_true, pixel_size_true)

viewer = napari.Viewer(ndisplay=3)

viewer.add_image(
    my_stack_norm,
    rendering='mip',
    name='volume',
    blending='translucent',
    opacity=1,
    colormap='inferno',
    scale=my_scale
)

for p, img in deconvolved.items():
    viewer.add_image(
        img,
        rendering='mip',
        name=f'{method} volume {p}',
        blending='translucent',
        opacity=1,
        colormap='inferno',
        scale=my_scale
    )

viewer.axes.visible = True
viewer.camera.angles = (45, 45, 45)
viewer.camera.zoom = 1

if __name__ == '__main__':
    napari.run()