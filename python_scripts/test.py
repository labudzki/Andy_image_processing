
from tifffile import imread, imwrite, TiffFile

stack = imread(
        '/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/Analysis/DeconvolvedData/260702/CFL2605A123/Run04/Stack_Run04_DL2_RL10.ome.tif'
    )

print(stack.shape)