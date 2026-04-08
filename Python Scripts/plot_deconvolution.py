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

# ---------------------
# Creating random noise - dont think i need this
# ---------------------
rng = np.random.default_rng()


# ---------------------
# Import 20X data 
# ---------------------
# astro = color.rgb2gray(data.astronaut())
path_movie = Path(
rf'/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/RawData/260408/CFL2601A049/Run01/Run01_MMStack_Pos0.ome.tif'
)

with TiffFile(path_movie) as tif:
    stack = tif.asarray()

print(stack.shape, stack.dtype)  
#%%
# # Select a specific slice from the 2D stack (e.g., slice index 10)
slice_index_t = 70 # first dim
# # slice_index_z = 9  # second dim
astro = stack[slice_index_t, :, :]



# ---------------------
# Import experimental PSF
# ---------------------
# psf = np.ones((5, 5)) / 25
psf = np.load('/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/260316/PSF_crops/PSF_output/avg_psf.npy')

# Normalize so it sums to 1 (required for RL deconvolution)
psf = psf / psf.sum()

astro = conv2(astro, psf, 'same')
# # Add Noise to Image
# astro_noisy = astro.copy()
# astro_noisy += (rng.poisson(lam=25, size=astro.shape) - 10) / 255.0

# Restore Image using Richardson-Lucy algorithm
deconvolved_RL = restoration.richardson_lucy(astro, psf, num_iter=30)
# deconvolved_RL = restoration.richardson_lucy(astro_noisy, psf, num_iter=30)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
plt.gray()

for a in (ax[0], ax[1]):
# for a in (ax[0], ax[1], ax[2]):
    a.axis('off')

ax[0].imshow(astro)
ax[0].set_title('Original Data')

# ax[1].imshow(astro_noisy)
# ax[1].set_title('Noisy data')

ax[1].imshow(deconvolved_RL, vmin=astro.min(), vmax=astro.max())
# ax[1].imshow(deconvolved_RL, vmin=astro_noisy.min(), vmax=astro_noisy.max())
ax[1].set_title('Restoration using\nRichardson-Lucy')


fig.subplots_adjust(wspace=0.02, hspace=0.2, top=0.9, bottom=0.05, left=0, right=1)
plt.show()
