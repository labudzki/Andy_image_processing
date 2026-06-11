
"""
Volume plane rendering
======================

Display one 3-D volume layer using the add_volume API and display it as a plane
with a simple widget for modifying plane parameters
"""
import napari
import numpy as np
from napari.utils.translations import trans
from skimage import data
from pathlib import Path
from tifffile import TiffFile
import xml.etree.ElementTree as ET

fnum = 16 #to define run number

path_movie = Path(
rf'/Users/andrealabudzki/Library/CloudStorage/OneDrive-AMOLF/Data/Multi photon tests/260512 Bruker/BrukerMultiphoton/Tom Demo 3/25x - Nile Red/No Water/ZSeries-11262025-1205-004/ZSeries-11262025-1205-004_Cycle00001_Ch1_000001.ome.tif'
# '/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/RawData/260216/CFL2601A042/Run01/Run01_MMStack_Pos0.ome.tif'
# '/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/RawData/251128/CFL2510A005/Run10/Run10_MMStack_Pos0.ome.tif'
)

# Load as a NumPy array (ensure it’s contiguous)
with TiffFile(path_movie) as tif:
    stack = tif.asarray()
    ome_metadata = tif.ome_metadata

# path_movie2 = Path(
# '/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/RawData/260430/CFL2603A076/Run03/Run03_MMStack_Pos0.ome.tif'
# )

# # Load as a NumPy array (ensure it’s contiguous)
# with TiffFile(path_movie2) as tif:
#     stack2 = tif.asarray()
#     ome_metadata2 = tif.ome_metadata

# ----------------------
# WIP: Obtain correct scaling for x, y, z
# ----------------------

# Obtaining z intervals from OME-XML metadata using wildcard search (ignoring namespaces)
root = ET.fromstring(ome_metadata)
planes = root.findall(".//{*}Plane")

# z_positions = []

z = root.get("z-step_um") # or p.get("PositionZ")
    # if z is not None:
    #     z_positions.append(float(z))

print(z)

# ----------------------

z_interval = 2  #um
# z_interval = 0.5 #um

# pixel_size = 6.5 #um
# magnification = 60  # adjust based on data
# pixel_size_true = pixel_size / magnification  # um
pixel_size_true = 512/352.77
print(f"Pixel size (true): {pixel_size_true} um")   


my_scale = (z_interval, pixel_size_true, pixel_size_true)

# stack0 = stack[0]  # take first timepoint
stack0 = stack
print(f"Stack shape: {stack0.shape}")
#%% Napari viewer

viewer = napari.Viewer(ndisplay=3)

volume_layer = viewer.add_image(
    stack0, 
    rendering='mip', 
    name='volume', 
    blending= 'translucent', #'opaque', # 'additive', 
    opacity=1,
    colormap = 'inferno', 
    # interpolation = 'spline36'
    # contrast_limits=[144, 258],
    # contrast_limits=[158, 337],
    scale=my_scale
)
z_interval2 = 1 #um
my_scale2 = (z_interval2, pixel_size_true, pixel_size_true)

# volume_layer2 = viewer.add_image(
#     stack2, 
#     rendering='mip', 
#     name='volume2', 
#     blending= 'translucent', #'opaque', # 'additive', 
#     opacity=1,
#     colormap = 'inferno', 
#     # interpolation = 'spline36'
#     # contrast_limits=[144, 258],
#     # contrast_limits=[158, 337],
#     scale=my_scale2
# )

# add the same volume and render as plane
# plane should be in 'additive' blending mode or depth looks all wrong
# plane_parameters = {
#     'position': (10, 326, 645),
#     # 'position': (32, 32, 32),
#     'normal': (0, 1, 0),
#     'thickness': 1,
#     # 'thickness': 10,
# }

# plane_layer = viewer.add_image(
#     stack0,
#     rendering='average',
#     name='plane',
#     depiction='plane',
#     blending='additive',
#     opacity=1,
#     plane=plane_parameters,
#     colormap = 'inferno',
#     contrast_limits=[113, 300],
#     scale = my_scale
#   )

viewer.axes.visible = True
viewer.camera.angles = (45, 45, 45)
viewer.camera.zoom = 1
viewer.text_overlay.text = trans._(
    """
 shift + click and drag to move the plane
press 'x', 'y' or 'z' to orient the plane along that axis around the cursor
press 'o' to orient the plane normal along the camera view direction
press and hold 'o' then click and drag to make the plane normal follow the camera
"""
)

viewer.text_overlay.visible = True

# These dont work for now

# viewer.window._qt_window.showMaximized()  # Maximizes the window

# @viewer.bind_key('c')
# def show_camera_angles(viewer):
#     print("Camera angles:", viewer.camera.angles)
# %% Run napari
if __name__ == '__main__':
    napari.run()
