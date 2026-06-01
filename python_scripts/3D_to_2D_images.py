# The purpose of this code is to select a slice from a 3D image array and save it as a 2D image file. you can view the 3D image in napari with the commented out portion
import napari
import numpy as np
from tifffile import imwrite, TiffFile
from pathlib import Path
# from LOCOMYCO_ImgAnalysis import tifreading as u_tif
import matplotlib.pyplot as plt

path_movie = Path(
# rf"c:\Users\labudzki\OneDrive - AMOLF\Desktop\Data\HSE2508A264\Run{fnum}\Run{fnum}_MMStack_Pos0.ome.tif"
# rf"C:\Users\labudzki\OneDrive - AMOLF\Documents\Repositories\confocal_processing\lipid movies\SAL2506A042\Mov21_MMStack_Pos0.ome.tif" #magnification 50X
# rf"C:\Users\labudzki\AMOLF-SHIMIZU Dropbox\DATA\Ach_data\5. Lipids and Organelles imaging\RawData\251125\CFL2510A002\Run7\Run7_MMStack_Pos0.ome.tif"
# rf"C:\Users\labudzki\AMOLF-SHIMIZU Dropbox\DATA\Ach_data\5. Lipids and Organelles imaging\RawData\251125\CFL2510A002\Run10\Run10_MMStack_Pos0.ome.tif"
# rf"C:\Users\labudzki\AMOLF-SHIMIZU Dropbox\DATA\Ach_data\5. Lipids and Organelles imaging\RawData\251128\CFL2510A005\Run19\Run19_MMStack_Pos0.ome.tif"
# rf"C:\Users\labudzki\AMOLF-SHIMIZU Dropbox\DATA\Ach_data\x. SetUp Charac\Point spread function\260316\PSF_20x_nanobeads680-605\PSF_20x_nanobeads680-605_MMStack_Pos0.ome.tif"
rf'/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/RawData/260408/CFL2601A049/Run03/Run03_MMStack_Pos0.ome.tif'
)

output_path = Path(rf"/Users/andrealabudzki/Desktop")

# Path to the Images folder in this repo
# images_dir = Path(__file__).resolve().parents[1] / "Images"

                  
                

# I want 9/20 and 20/39 

# Load as a NumPy array 
with TiffFile(path_movie) as tif:
    stack = tif.asarray()

print(stack.shape, stack.dtype)  

# Select a specific slice from the 3D stack (e.g., slice index 10)
slice_index_t = 130 # first dim
# slice_index_z = 9  # second dim
output_stack = stack[slice_index_t, :, :]
# output_stack = stack[slice_index_t, slice_index_z, :, :]
print(output_stack.shape, output_stack.dtype)

# # Print image
# plt.figure(figsize=(12, 5))
# plt.imshow(output_stack)
# plt.title('2D image')
# # plt.savefig(images_dir / f"251128_CFL2510A005_Run19_2D_slice_t{slice_index_t}_z{slice_index_z}.png", dpi=300)
# plt.show()

# imwrite(output_path / f"t{slice_index_t}.tif", output_stack)

imwrite(output_path / f"t{slice_index_t}.png", output_stack)

# # Save the selected slice as a 2D image file
# output_file = output_path / f"Run19_t{slice_index_t}_z{slice_index_z}.tif"
# imwrite(output_file, output_stack.astype(np.uint16))       
# print(f"Saved 2D image slice to {output_file}")
    

# #stack=u_tif.load_tiff_stack(path_movie, squeeze=True)

# stack = np.array(stack, dtype='float32')  # reduce memory usage



# viewer = napari.Viewer()

# viewer.add_image(stack, name='Raw video', multiscale=False, axis_labels=['Time','Z','Y','X'])
# # viewer.add_image(stack, name='Raw video', multiscale=False, axis_labels=['Z','Y','X']) # for Run1 and Run2
# # viewer.add_image(stack, name='Raw video', multiscale=False, axis_labels=['Time','Y','X']) # for Run5
# # #print(viewer.layers.data
# napari.run()


