"""
=====================
Image Deconvolution
=====================
In this example, we deconvolve an image using Richardson-Lucy
deconvolution algorithm ([1]_, [2]_).

The algorithm is based on a PSF (Point Spread Function),
where PSF is described as the impulse response of the
optical system. The blurred image is sharpened through a number of
iterations, which needs to be hand-tuned.

.. [1] William Hadley Richardson, "Bayesian-Based Iterative
       Method of Image Restoration",
       J. Opt. Soc. Am. A 27, 1593-1607 (1972), :DOI:`10.1364/JOSA.62.000055`

.. [2] https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imwrite, TiffFile
from pathlib import Path
from scipy.signal import convolve2d as conv2
from skimage import color, data, restoration
from PIL import Image
from datetime import datetime

# ---------------------
# Creating random noise 
# ---------------------
# rng = np.random.default_rng()


# ---------------------
# Import 20X data 
# ---------------------
# astro = color.rgb2gray(data.astronaut())
path_movie = Path(
    #20 x
rf'/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/RawData/260408/CFL2601A049/Run01/Run01_MMStack_Pos0.ome.tif'
    # 60 x
# rf'/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/RawData/260408/CFL2601A049/Run02/Run02_MMStack_Pos0.ome.tif'
# rf'/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/RawData/260408/CFL2601A049/Run03/Run03_MMStack_Pos0.ome.tif'
# rf'/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/RawData/250721/SAL2506A042/Mov20/Mov20_MMStack_Pos0.ome.tif'
# rf'/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/RawData/260401/CFL2601A058/Run07/Run07_MMStack_Pos0.ome.tif'
# rf'/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/RawData/251128/CFL2510A005/Run4/Run4_MMStack_Pos0.ome.tif'
)

with TiffFile(path_movie) as tif:
    stack = tif.asarray()

# Select a specific slice from the 2D stack (e.g., slice index 10)
slice_index_t = 70 # first dim
astro = stack[slice_index_t, :, :]

print(astro.shape, astro.dtype, astro.min(), astro.max())

# Subtract background before normalizing
astro = astro - astro.min()  # or use a proper background ROI if you have one

# Normalize
astro = astro / astro.max()
#%%




# ---------------------
# Import experimental PSF
# ---------------------
# psf = np.ones((5, 5)) / 25

#20x PSF
psf = np.load('/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/20x_objective/PSF_crops/PSF_output_20260410_1045/avg_psf.npy')

#60x PSF
# psf = np.load('/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/60x_objective/PSF_crops/PSF_output_20260413_0932/avg_psf.npy')
# Normalize so it sums to 1 (required for RL deconvolution)
psf = psf / psf.sum()

print("image slice shape:", astro.shape)  # should be (712, 1008)
print("PSF shape:", psf.shape)            # should also be 2D

# Artificial blurring of the image using convolution with the PSF
# astro = conv2(astro, psf, 'same')
# # Add Noise to Image
# astro_noisy = astro.copy()
# astro_noisy += (rng.poisson(lam=25, size=astro.shape) - 10) / 255.0

# # Restore Image using Richardson-Lucy algorithm
# deconvolved_RL = restoration.richardson_lucy(astro, psf, num_iter=30)
# # deconvolved_RL = restoration.richardson_lucy(astro_noisy, psf, num_iter=30)

# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
# plt.gray()

# for a in (ax[0], ax[1]):
# # for a in (ax[0], ax[1], ax[2]):
#     a.axis('off')

# ax[0].imshow(astro)
# ax[0].set_title('Original Data')

# # ax[1].imshow(astro_noisy)
# # ax[1].set_title('Noisy data')

# ax[1].imshow(deconvolved_RL, vmin=astro.min(), vmax=astro.max())
# # ax[1].imshow(deconvolved_RL, vmin=astro_noisy.min(), vmax=astro_noisy.max())
# ax[1].set_title('Restoration using\nRichardson-Lucy')


# fig.subplots_adjust(wspace=0.02, hspace=0.2, top=0.9, bottom=0.05, left=0, right=1)
# plt.show()

# Restore Image using Richardson-Lucy algorithm at different iterations
deconvolved_10 = restoration.richardson_lucy(astro, psf, num_iter=10)
deconvolved_20 = restoration.richardson_lucy(astro, psf, num_iter=20)
deconvolved_30 = restoration.richardson_lucy(astro, psf, num_iter=30)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))
plt.gray()

for a in ax:
    a.axis('off')

ax[0].imshow(astro)
ax[0].set_title('Original Data')

ax[1].imshow(deconvolved_10, vmin=astro.min(), vmax=astro.max())
ax[1].set_title('RL – 10 iterations')

# ax[2].imshow(deconvolved_20, vmin=astro.min(), vmax=astro.max())
# ax[2].set_title('RL – 20 iterations')

# ax[3].imshow(deconvolved_30, vmin=astro.min(), vmax=astro.max())
# ax[3].set_title('RL – 30 iterations')

fig.subplots_adjust(wspace=0.02, hspace=0.2, top=0.9, bottom=0.05, left=0, right=1)
plt.show()

# # Save deconvolved image
# save_path = Path(
#     rf'/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/Analysis/251128/CFL2510A005/Run5/Run5_t{slice_index_t}_deconvolved_RL.tif'
# )
# deconvolved_uint16 = (deconvolved_30 * 65535).astype(np.uint16) #the image is normalized between 0 and 1, so we scale it to the uint16 range
# imwrite(save_path, deconvolved_uint16)

# astro_uint8 = (astro * 255).astype(np.uint8)
# deconvolved_uint8 = (deconvolved_30 * 255).astype(np.uint8)


# save_path_original_png = Path(
#     rf'/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/Analysis/251128/CFL2510A005/Run5/Run5_t{slice_index_t}_original.png'
# )

# save_path_deconvolved_png = Path(
#     rf'/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/Analysis/251128/CFL2510A005/Run5/Run5_t{slice_index_t}_deconvolved_RL.png'
# )
# Image.fromarray(astro_uint8).save(save_path_original_png)
# Image.fromarray(deconvolved_uint8).save(save_path_deconvolved_png)

# Save deconvolved images with timestamp as tifs and pngs

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = Path(rf'/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/Analysis/251128/CFL2510A005/Run4')
save_dir = Path(rf'/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/Analysis/260408/CFL2601A049/Run01')
# 260408/CFL2601A049/Run01/Run01_MMStack_Pos0.ome.tif

astro_uint8 = (astro * 255).astype(np.uint8)

# Save original
Image.fromarray(astro_uint8).save(
    save_dir / f"Run01_t{slice_index_t}_original_{timestamp}.png"
)

# Save each deconvolved iteration as uint16 tif and uint8 png
for n_iter, deconvolved in zip([10, 20, 30], [deconvolved_10, deconvolved_20, deconvolved_30]):
    
    # PNG
    deconvolved_uint8 = (deconvolved * 255).astype(np.uint8)
    Image.fromarray(deconvolved_uint8).save(
        save_dir / f"Run01_t{slice_index_t}_deconvolved_RL_{n_iter}iter_{timestamp}.png"
    )
    
    # TIF (uint16, only for 30 iter as before — or keep all if you want)
    if n_iter == 30:
        deconvolved_uint16 = (deconvolved * 65535).astype(np.uint16)
        imwrite(save_dir / f"Run01_t{slice_index_t}_deconvolved_RL_{n_iter}iter_{timestamp}.tif", deconvolved_uint16)
