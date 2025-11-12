import napari
import numpy as np

dummy_stack = np.random.rand(10, 50, 50)  # simple 3D array

viewer = napari.Viewer()
viewer.add_image(dummy_stack, name='dummy')
napari.run()