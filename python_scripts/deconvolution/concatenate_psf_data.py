
"""
Concatenating PSF data
======================

When capturing PSF data, it's easy to run out of memory due to the large range and fine sampling of the z stacks. So I captured the same FOV with various z ranges. 
With this code, I concatenate the PSF data into a single 3D stack, getting rid of the overlap, and then save it as a tif file.

As of now this doesnt seem to be working correctly. When I open the concatenated stack in imageJ, the first stacks are black for a long time and then I get some beads. Different from what I'd expect when looking at Runs 12 and 14 separately.
"""



import numpy as np
from tifffile import TiffFile, imwrite
from pathlib import Path
import xml.etree.ElementTree as ET


# ---------------------
# Import 60X 3D data 
# ---------------------
path_movie1 = Path(
'/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/3d/260706_Run12/Stack_Run12_170611_MMStack_Pos0.ome.tif'
)

path_movie2 = Path(
'/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/Point spread function/3d/260706_Run14/Stack_Run14_170739_MMStack_Pos0.ome.tif'
)

with TiffFile(path_movie1) as tif:
    stack1 = tif.asarray()

with TiffFile(path_movie2) as tif:
    stack2 = tif.asarray()

print('Stack 1 shape and dtype:', stack1.shape, stack1.dtype)
print('Stack 2 shape and dtype:', stack2.shape, stack2.dtype)

# z int 1: -15 - 18 um, sawtooth data, 0.1 um z interval
# z int 2: -10 - 23 um, sawtooth data, 0.1 um z interval

# Find the overlap between the two stacks and concatenate them 

z1 = np.linspace(-15, 18, stack1.shape[0])  # 331 points, 0.1 um step
z2 = np.linspace(-10, 23, stack2.shape[0])  # 331 points, 0.1 um step

# sanity check the step size is really 0.1 um
assert np.isclose(z1[1] - z1[0], 0.1)
assert np.isclose(z2[1] - z2[0], 0.1)

all_z = np.concatenate([z1, z2])
all_frames = np.concatenate([stack1, stack2], axis=0)

order = np.argsort(all_z)
all_z_sorted = all_z[order]
all_frames_sorted = all_frames[order]

# drop duplicates in the overlap region (-10 to 18 um)
tol = 0.05  # half the 0.1 um step
keep = np.ones(len(all_z_sorted), dtype=bool)
last_z = -np.inf
for i, z in enumerate(all_z_sorted):
    if z - last_z < tol:
        keep[i] = False
    else:
        last_z = z

merged_stack = all_frames_sorted[keep]
merged_z = all_z_sorted[keep]

print('Merged stack shape:', merged_stack.shape, 'Z range:', merged_z.min(), merged_z.max()) 

# Save the merged stack as a tif file

pixel_size_camera = 6.5  # um
mag = 60
pixel_size_xy = pixel_size_camera / mag  # um

output_path = path_movie1.parent / 'Run12_Run14_merged.tif'

imwrite(
    output_path,
    merged_stack,
    imagej=True,
    resolution=(1 / pixel_size_xy, 1 / pixel_size_xy),
    metadata={
        'axes': 'ZYX',
        'unit': 'um',
        'spacing': 0.1,  # z-step in um
    },
)

print(f"Saved merged stack to {output_path}")
print(merged_stack.shape, merged_stack.dtype)