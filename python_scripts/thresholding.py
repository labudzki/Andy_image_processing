# %% Imports
# ----------------------------------------------------------------------
import napari
import numpy as np
from tifffile import TiffFile
from PIL import Image 
from pathlib import Path
from LOCOMYCO_ImgAnalysis import tifreading as u_tif
import matplotlib.pyplot as plt
import skimage as ski
import os

#blur the image with a gaussian filter, then threshold, then apply mask to original image, filter for component size to remove bigger artifacts in the background
# convert to automatic thresholding (Otsu's method)?

#%% Upload image, perform gaussian blur, print histograms
# ----------------------------------------------------------------------

# Uploading image
path_movie = Path(
rf"C:\Users\labudzki\AMOLF-SHIMIZU Dropbox\DATA\Ach_data\5. Lipids and Organelles imaging\Analysis\251128\CFL2510A005\Run5\251128_CFL2510A005_Run5_noscalebar.png"
)

output_dir = Path(rf"C:\Users\labudzki\AMOLF-SHIMIZU Dropbox\DATA\Ach_data\5. Lipids and Organelles imaging\Analysis\251128\CFL2510A005\Run5")

# Load as a NumPy array (ensure it’s contiguous/continuous)
if path_movie.suffix.lower() in [".tif", ".tiff"]:
    with TiffFile(path_movie) as tif:
        stack = tif.asarray()

elif path_movie.suffix.lower() == ".png":
    img = Image.open(path_movie)
    stack = np.asarray(img)
    #convert to grayscale if it's RGB
    if stack.ndim == 3 and stack.shape[-1] == 3:
        stack = stack.mean(axis=-1) 

else:
    raise ValueError("Unsupported file format")
    
# # Select a single z and t for now
# slice_index_t = 7 # first dim
# slice_index_z = 7  # second dim
# stack = stack[slice_index_t, slice_index_z, :, :]
print(stack.shape)

# stack = np.array(stack, dtype='float32')  # reduce memory usage

# blur to remove noise
blurred_stack = ski.filters.gaussian(stack, sigma=1.0, preserve_range=True)

# Visualize original and blurred images along with their histograms
# plt.figure(figsize=(12, 5))

# Top-left: original image
# plt.subplot(2, 2, 1)
fname = "original_image.png"
plt.figure(figsize=(12, 5))
plt.imshow(stack, cmap='gray')
plt.title('Initial image')
plt.axis('off')
# plt.savefig(os.path.join(output_dir, fname), dpi=600, bbox_inches="tight")
plt.show()

# Top-right: blurred image
# plt.subplot(2, 2, 2)
plt.figure(figsize=(12, 5))
plt.imshow(blurred_stack, cmap='gray')
plt.title('Blurred image')
plt.axis('off')

# Bottom-left: histogram of original
# plt.subplot(2, 2, 3)
plt.figure(figsize=(12, 5))
plt.hist(stack.ravel(), bins=300, color='blue', alpha=0.7) #ravel flattens the array for histogram without modifying the original data
plt.title('Histogram of Original Image Intensities')
plt.xlabel('Intensity Value')
plt.ylabel('Frequency')
#%%

plt.figure(figsize=(12, 5))
plt.hist(stack.ravel(), bins=300, color='blue', alpha=0.7)
plt.title('Histogram of Non-Zero Image Intensities')
plt.xlabel('Intensity Value')
plt.ylabel('Frequency')
plt.xlim(10, 300) #limit x-axis to better visualize the distribution of non-zero pixels
plt.ylim(0, 10000)
plt.show()
#%%
# Bottom-right: histogram of blurred
# plt.subplot(2, 2, 4)

plt.figure(figsize=(12, 5))
plt.hist(blurred_stack.ravel(), bins=300, color='blue', alpha=0.7)
plt.title('Histogram of Blurred Image Intensities')
plt.xlabel('Intensity Value')
plt.ylabel('Frequency')
# plt.tight_layout()
# fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
plt.show()

# # Find max frequency bin and value in original image and thresholded image
# hist, bin_edges = np.histogram(stack.flatten(), bins=100)
# max_freq_index = np.argmax(hist)
# max_freq_bin = bin_edges[max_freq_index]
# print(f"Max frequency bin in original image: {max_freq_bin}, Frequency: {hist[max_freq_index]}")

# hist_thresh, bin_edges_thresh = np.histogram(stack_thresh.flatten(), bins=100)
# max_freq_index_thresh = np.argmax(hist_thresh)
# max_freq_bin_thresh = bin_edges_thresh[max_freq_index_thresh]
# print(f"Max frequency bin in thresholded image: {max_freq_bin_thresh}, Frequency: {hist_thresh[max_freq_index_thresh]}") #all background pixels set to 0, so the 0 bin has the highest frequency  

#%% Create mask for selecting hyphae only
# ----------------------------------------------------------------------
t_bg = 16 # 180 #0.003 - value for when gaussian blur normalizes intensities between 0 and 1
binary_mask = blurred_stack > t_bg

# perform automatic thresholding - for now Otsu gives a threshold that is too high and removes a significant portion of the hyphae 
t_otsu = ski.filters.threshold_otsu(blurred_stack)
binary_mask_otsu = blurred_stack > t_otsu
print("Otsu threshold bg t = {}.".format(t_otsu))

 # clean up mask with morphological operations
 # filter for component size to remove bigger artifacts in the background
binary_mask_cleaned = ski.morphology.remove_small_objects(binary_mask, min_size = 2000, connectivity=2)
# binary_mask_cleaned = binary_mask 
# print( "skipped removing of small objects")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(binary_mask, cmap="gray")
plt.title('Binary mask manual')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(binary_mask_cleaned, cmap="gray")
plt.title(f'Binary mask small objects removed')
plt.axis('off')

plt.tight_layout()
plt.show()

plt.figure()
plt.imshow(binary_mask_otsu, cmap="gray")
plt.title('Binary mask otsu')
plt.axis('off')
plt.show()


#%% Apply manual mask to original image
selection = stack.copy()
selection[~binary_mask_cleaned] = 0

fname = "masked_image.png"
plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
plt.imshow(selection, cmap='gray')
plt.title('Image after applying bg mask')
plt.axis('off')
# plt.savefig(os.path.join(output_dir, fname), dpi=600, bbox_inches="tight")
plt.show()

# plt.subplot(1, 2, 2)
plt.figure(figsize=(12, 5))
nonzero_vals = selection[selection != 0] #remove zeros for better visualization of intensity values 
plt.hist(nonzero_vals.ravel(), bins=300, color='blue', alpha=0.7)
plt.title('Histogram of masked image, ignoring zeroes')
plt.xlabel('Intensity Value')
plt.ylabel('Frequency')
plt.show()

#%% Create mask for fluorescence signal above 
# ----------------------------------------------------------------------    
t_fluo = 130 #this seems to be too high, change to slightly lower value
binary_mask_fluo = selection > t_fluo
# binary_mask_cleaned = ski.morphology.remove_small_objects(binary_mask, min_size = 1000, connectivity=2)

#Otsu thresholding for fluorescence - not working well here either, threshold way too low, we need to be more selective for fluo 
t_otsu_fluo = ski.filters.threshold_otsu(selection[selection>0]) #only consider non-zero pixels for finding threshold
binary_mask_fluo_otsu = selection > t_otsu_fluo
print("Otsu threshold fluo t = {}.".format(t_otsu_fluo))

# Calculate lipid volume %
# ----------------------------------------------------------------------
num_fluo_pixels = np.sum(binary_mask_fluo)    
num_fluo_pixels_otsu = np.sum(binary_mask_fluo_otsu)  
num_hyphae_pixels = np.sum(binary_mask_cleaned)
lipid_volume_percent = (num_fluo_pixels / num_hyphae_pixels) * 100
lipid_volume_percent_otsu = (num_fluo_pixels_otsu / num_hyphae_pixels) * 100
print(f"Num of fluorescent pixels: {num_fluo_pixels}")
print(f"Num of fluorescent pixels otsu: {num_fluo_pixels_otsu}")
print(f"Total num hyphae pixels: {num_hyphae_pixels}")
print(f"Lipid volume percentage: {lipid_volume_percent:.2f}%")
print(f"Lipid volume percentage otsu: {lipid_volume_percent_otsu:.2f}%")

plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.imshow(selection, cmap="gray")
# plt.title('Image after applying bg mask')
# plt.axis('off')
fname = "fluorescence_mask_manual.png"
# plt.subplot(1, 2, 1)
plt.imshow(binary_mask_fluo, cmap="gray")
plt.title('Binary mask manual, lipid volume: {:.0f}%'.format(lipid_volume_percent), fontsize = 24)
plt.axis('off')
plt.savefig(os.path.join(output_dir, fname), dpi=800, bbox_inches="tight")
plt.show()

fname = "fluorescence_mask_otsu.png"
# plt.subplot(1, 2, 2)
plt.imshow(binary_mask_fluo_otsu, cmap="gray")
plt.title('Binary mask Otsu, lipid volume: {:.0f}%'.format(lipid_volume_percent_otsu), fontsize = 16)
plt.axis('off')
plt.savefig(os.path.join(output_dir, fname), dpi=800, bbox_inches="tight")
plt.show()

# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, fname), dpi=600, bbox_inches="tight")




#
# #%%Run napari
# viewer = napari.Viewer()
# # viewer.theme = 'light' #Set background to white so i can see cutting of background more easily
# viewer.add_image(stack, name='Raw img', multiscale=False, axis_labels=['Y','X']) 
# # viewer.add_image(stack, name='Raw video', multiscale=False, axis_labels=['Time','Z','Y','X'])
# viewer.add_image(mask_white, name='Mask', multiscale=False, axis_labels=['Y','X']) 
# viewer.add_image(mask_fluorescence, name='Fluorescence mask', multiscale=False, axis_labels=['Y','X']) 

# napari.run()


# set threshold for staining

# # of pixels above threshold


# %%
