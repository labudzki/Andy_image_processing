import napari
import numpy as np
from tifffile import TiffFile
from pathlib import Path
from LOCOMYCO_ImgAnalysis import tifreading as u_tif

fnum = 3

path_movie = Path(
# rf"c:\Users\labudzki\OneDrive - AMOLF\Desktop\Data\HSE2508A264\Run{fnum}\Run{fnum}_MMStack_Pos0.ome.tif"
# rf"C:\Users\labudzki\OneDrive - AMOLF\Documents\Repositories\confocal_processing\lipid movies\SAL2506A042\Mov21_MMStack_Pos0.ome.tif" #magnification 50X
# rf"C:\Users\labudzki\AMOLF-SHIMIZU Dropbox\DATA\Ach_data\5. Lipids and Organelles imaging\RawData\251125\CFL2510A002\Run7\Run7_MMStack_Pos0.ome.tif"
# rf"C:\Users\labudzki\AMOLF-SHIMIZU Dropbox\DATA\Ach_data\5. Lipids and Organelles imaging\RawData\251125\CFL2510A002\Run10\Run10_MMStack_Pos0.ome.tif"
rf"C:\Users\labudzki\AMOLF-SHIMIZU Dropbox\DATA\Ach_data\5. Lipids and Organelles imaging\RawData\251128\CFL2510A005\Run5\Run5_MMStack_Pos0.ome.tif"
)

# Load as a NumPy array (ensure it’s contiguous)
with TiffFile(path_movie) as tif:
    stack = tif.asarray()
    

#stack=u_tif.load_tiff_stack(path_movie, squeeze=True)

stack = np.array(stack, dtype='float32')  # reduce memory usage

print(stack.shape, stack.dtype)  # sanity check


viewer = napari.Viewer()

# viewer.add_image(stack, name='Raw video', multiscale=False, axis_labels=['Time','Z','Y','X'])
# viewer.add_image(stack, name='Raw video', multiscale=False, axis_labels=['Z','Y','X']) # for Run1 and Run2
viewer.add_image(stack, name='Raw video', multiscale=False, axis_labels=['Time','Y','X']) # for Run5
# #print(viewer.layers.data
napari.run()


