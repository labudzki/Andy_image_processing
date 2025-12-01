# %% Initizialization
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

viewer = napari.Viewer(ndisplay=3)

fnum = 16 #to define run number

path_movie = Path(
rf"c:\Users\labudzki\OneDrive - AMOLF\Desktop\Data\HSE2508A264\Run{fnum}\Run{fnum}_MMStack_Pos0.ome.tif"
# rf"C:\Users\labudzki\OneDrive - AMOLF\Documents\Repositories\confocal_processing\lipid movies\SAL2506A042\Mov21_MMStack_Pos0.ome.tif" #magnification 50X
# rf"C:\Users\labudzki\AMOLF-SHIMIZU Dropbox\DATA\Ach_data\5. Lipids and Organelles imaging\RawData\251125\CFL2510A002\Run{fnum}\Run{fnum}_MMStack_Pos0.ome.tif"
# rf"C:\Users\labudzki\AMOLF-SHIMIZU Dropbox\DATA\Ach_data\5. Lipids and Organelles imaging\RawData\281125\CFL2510A005\Run17\Run17_MMStack_Pos0.ome.tif"
# rf"C:\Users\labudzki\AMOLF-SHIMIZU Dropbox\DATA\Ach_data\5. Lipids and Organelles imaging\RawData\281125\CFL2510A005\Run5\Run5_MMStack_Pos0.ome.tif"
# rf"C:\Users\labudzki\AMOLF-SHIMIZU Dropbox\DATA\Ach_data\5. Lipids and Organelles imaging\RawData\251125\CFL2510A002\Run10\Run10_MMStack_Pos0.ome.tif"
)

# Load as a NumPy array (ensure it’s contiguous)
with TiffFile(path_movie) as tif:
    stack = tif.asarray()
    ome_metadata = tif.ome_metadata

#Obtain pixel sizes 
voxel_size_z = float(ome_metadata.split('PhysicalSizeZ="')[1].split('"')[0])
voxel_size_y = float(ome_metadata.split('PhysicalSizeY="')[1].split('"')[0])
# voxel_size_x = float(ome_metadata.split('PhysicalSizeX="')[1].split('"')[0])
print(ome_metadata)
# print(f"Voxel sizes: Z={voxel_size_z} um, Y={voxel_size_y} um, X={voxel_size_x} um")

# scale = [voxel_size_z, voxel_size_y, voxel_size_x]
z_scale = 1 * voxel_size_z  # adjust for better visualization
my_scale = (5*voxel_size_z/voxel_size_y, 1, 1)

# stack0 = stack[0]  # take first timepoint
stack0 = stack
print(f"Stack shape: {stack0.shape}")

# Sample 3D data from the original code
# # add a volume
# blobs = data.binary_blobs(
#     length=64, volume_fraction=0.1, n_dim=3
# ).astype(np.float32)

volume_layer = viewer.add_image(
    stack0, 
    rendering='mip', 
    name='volume', 
    blending='additive', 
    opacity=1,
    colormap = 'inferno', 
    # contrast_limits=[144, 258],
    # contrast_limits=[158, 337],
    scale=my_scale
)

# add the same volume and render as plane
# plane should be in 'additive' blending mode or depth looks all wrong
plane_parameters = {
    'position': (10, 326, 645),
    # 'position': (32, 32, 32),
    'normal': (0, 1, 0),
    'thickness': 1,
    # 'thickness': 10,
}

plane_layer = viewer.add_image(
    stack0,
    rendering='average',
    name='plane',
    depiction='plane',
    blending='additive',
    opacity=1,
    plane=plane_parameters,
    colormap = 'inferno',
    contrast_limits=[113, 300],
    scale = my_scale
  )

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
