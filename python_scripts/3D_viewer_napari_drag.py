import napari
import numpy as np
from tifffile import TiffFile

viewer = napari.Viewer()
napari.run()

# def my_reader(path: str):
#     if not path.lower().endswith((".tif", ".tiff")):
#         return None

#     with TiffFile(path) as tif:
#         arr = tif.asarray().astype('float32')

#     return [(arr, {"name": path, "axis_labels": ['T','Z','Y','X']}, "image")]

# viewer = napari.Viewer()
# viewer.window.file_drop.connect(my_reader)

# napari.run()