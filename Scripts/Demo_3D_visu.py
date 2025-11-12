import napari
import numpy as np
from tifffile import TiffFile
from pathlib import Path
import utils.tifreading as u_tif

path_movie = Path(
    r"D:\AMOLF-SHIMIZU Dropbox\DATA\Ach_data\5. Lipids and Organelles imaging\RawData\250711\TES2505A017\Run15\Run15_MMStack_Pos0.ome.tif"
)

# Load as a NumPy array (ensure it’s contiguous)
with TiffFile(path_movie) as tif:
    stack = tif.asarray()
    

stack=u_tif.load_tiff_stack(path_movie, squeeze=True)

stack = np.array(stack, dtype='float32')  # reduce memory usage

print(stack.shape, stack.dtype)  # sanity check


viewer = napari.Viewer()
viewer.add_image(stack, name='Raw video', multiscale=False, axis_labels=['Time','Y','X'])
napari.run()
