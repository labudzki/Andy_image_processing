
#%%
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imwrite, TiffFile
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
# astro = color.rgb2gray(data.astronaut())
path_movie = Path(
# rf'/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/RawData/251128/CFL2510A005/Run10/Run10_MMStack_Pos0.ome.tif'
rf'/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/RawData/260301/CFL2601A045/Run04/Run04_MMStack_Pos0.ome.tif'
)

with TiffFile(path_movie) as tif:
    stack = tif.asarray()
print(stack.shape)

#%%

z_int_data = 1 #um
# Select a specific time point from the 3D stack 
slice_index_t = 8 # first time point
slice_index_z = 8
stack_single_timeframe = stack[slice_index_t, slice_index_z, :, :]

print(stack_single_timeframe.shape)

# Subtract background before normalizing
stack_single_timeframe = stack_single_timeframe - stack_single_timeframe.min()  # or use a proper background ROI if you have one

# Normalize
astro = stack_single_timeframe / stack_single_timeframe.max()
#%%

# ---------------------
# Import experimental PSF
# ---------------------

#60x WI axial PSF
psf = np.load('/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/PSF/Nikon_CFI_PlanApo_VC_60X_WI/psf_3d.npy')
z_int_psf = 0.2 #um


# need to downsample the psf to match the z interval of the data
psf_resampled = zoom(
    psf,
    zoom=(z_int_psf / z_int_data, 1, 1),
    order=1
)

# Normalize to 1
psf_resampled_norm = psf_resampled / psf_resampled.sum()

print("image slice shape:", astro.shape)  
print("PSF shape:", psf.shape) 
print("PSF resampled shape:", psf_resampled_norm.shape)            

psf_3d = psf_resampled_norm

#%%
n_iter = 10
# Restore Image using Richardson-Lucy algorithm at different iterations
deconvolved_10 = restoration.richardson_lucy(astro, psf_3d, num_iter=n_iter)
deconvolved_20 = restoration.richardson_lucy(astro, psf_3d, num_iter=n_iter+10)
deconvolved_30 = restoration.richardson_lucy(astro, psf_3d, num_iter=n_iter+20)
print(deconvolved_10.shape)
# -------------
# plotting - 2D
# -------------

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(16, 5))
plt.gray()

for a in ax:
    a.axis('off')

z_plane_to_show = 10

ax[0].imshow(astro[z_plane_to_show, :, :], vmin=astro.min(), vmax=astro.max())
ax[0].set_title('Original Data (xy plane)')

ax[1].imshow(deconvolved_10[z_plane_to_show, :, :], vmin=astro.min(), vmax=astro.max())
ax[1].set_title(rf'RL – {n_iter} iterations (xy plane)')

ax[2].imshow(deconvolved_20[z_plane_to_show, :, :], vmin=astro.min(), vmax=astro.max())
ax[2].set_title(rf'RL – {n_iter + 10} iterations (xy plane)')

# ax[3].imshow(deconvolved_30[z_plane_to_show, :, :], vmin=astro.min(), vmax=astro.max())
# ax[3].set_title(rf'RL – {n_iter + 20} iterations')

fig.subplots_adjust(wspace=0.02, hspace=0.2, top=0.9, bottom=0.05, left=0, right=1)
plt.show()

#%%
# -------------
# plotting - planes
# -------------

y_plane_to_show = astro.shape[1] // 2  # center Y

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(16, 5))
plt.gray()

for a in ax:
    a.axis('off')


ax[0].imshow(astro[:, y_plane_to_show, :], vmin=astro.min(), vmax=astro.max())
ax[0].set_title('Original Data (xz plane)')

ax[1].imshow(deconvolved_10[:, y_plane_to_show, :], vmin=astro.min(), vmax=astro.max())
ax[1].set_title(rf'RL – {n_iter} iterations (xz plane)')

ax[2].imshow(deconvolved_20[:, y_plane_to_show, :], vmin=astro.min(), vmax=astro.max())
ax[2].set_title(rf'RL – {n_iter + 10} iterations (xz plane)')

# ax[3].imshow(deconvolved_30[z_plane_to_show, :, :], vmin=astro.min(), vmax=astro.max())
# ax[3].set_title(rf'RL – {n_iter + 20} iterations')

fig.subplots_adjust(wspace=0.02, hspace=0.2, top=0.9, bottom=0.05, left=0, right=1)
plt.show()

#%%
x_plane_to_show = astro.shape[2] // 2  # center Y

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(16, 5))
plt.gray()

for a in ax:
    a.axis('off')


ax[0].imshow(astro[:, :, x_plane_to_show], vmin=astro.min(), vmax=astro.max())
ax[0].set_title('Original Data (yz plane)')

ax[1].imshow(deconvolved_10[:, :, x_plane_to_show], vmin=astro.min(), vmax=astro.max())
ax[1].set_title(rf'RL – {n_iter} iterations (yz plane)')

ax[2].imshow(deconvolved_20[:, :, x_plane_to_show], vmin=astro.min(), vmax=astro.max())
ax[2].set_title(rf'RL – {n_iter + 10} iterations (yz plane)')

# ax[3].imshow(deconvolved_30[z_plane_to_show, :, :], vmin=astro.min(), vmax=astro.max())
# ax[3].set_title(rf'RL – {n_iter + 20} iterations')

fig.subplots_adjust(wspace=0.02, hspace=0.2, top=0.9, bottom=0.05, left=0, right=1)
plt.show()


#%%
# -------------
# plotting - 3D
# -------------

pixel_size = 6.5 #um
magnification = 60  # adjust based on data
pixel_size_true = pixel_size / magnification  # um
# # pixel_size_true = 512/352.77
# print(f"Pixel size (true): {pixel_size_true} um")   

my_scale = (z_int_data, pixel_size_true, pixel_size_true)
viewer = napari.Viewer(ndisplay=3)

volume_layer = viewer.add_image(
    astro, 
    rendering='mip', 
    name='volume', 
    blending= 'translucent', #'opaque', # 'additive', 
    opacity=1,
    colormap = 'inferno', 
    scale=my_scale
)

volume_layer2 = viewer.add_image(
    deconvolved_10, 
    rendering='mip', 
    name='deconvolved volume', 
    blending= 'translucent', #'opaque', # 'additive', 
    opacity=1,
    colormap = 'inferno', 
    scale=my_scale
)

volume_layer3 = viewer.add_image(
    astro - deconvolved_10, 
    rendering='mip', 
    name='difference', 
    blending= 'translucent', #'opaque', # 'additive', 
    opacity=1,
    colormap = 'inferno', 
    scale=my_scale
)

viewer.axes.visible = True
viewer.camera.angles = (45, 45, 45)
viewer.camera.zoom = 1

 # Run napari
if __name__ == '__main__':
    napari.run()


# # Save deconvolved images with timestamp as tifs and pngs

# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# save_dir = Path(rf'/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/Analysis/251128/CFL2510A005/Run4')
# save_dir = Path(rf'/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/Analysis/260408/CFL2601A049/Run01')
# # 260408/CFL2601A049/Run01/Run01_MMStack_Pos0.ome.tif

# astro_uint8 = (astro * 255).astype(np.uint8)

# # Save original
# Image.fromarray(astro_uint8).save(
#     save_dir / f"Run01_t{slice_index_t}_original_{timestamp}.png"
# )

# # Save each deconvolved iteration as uint16 tif and uint8 png
# for n_iter, deconvolved in zip([10, 20, 30], [deconvolved_10, deconvolved_20, deconvolved_30]):
    
#     # PNG
#     deconvolved_uint8 = (deconvolved * 255).astype(np.uint8)
#     Image.fromarray(deconvolved_uint8).save(
#         save_dir / f"Run01_t{slice_index_t}_deconvolved_RL_{n_iter}iter_{timestamp}.png"
#     )
    
#     # TIF (uint16, only for 30 iter as before — or keep all if you want)
#     if n_iter == 30:
#         deconvolved_uint16 = (deconvolved * 65535).astype(np.uint16)
#         imwrite(save_dir / f"Run01_t{slice_index_t}_deconvolved_RL_{n_iter}iter_{timestamp}.tif", deconvolved_uint16)

# %%
