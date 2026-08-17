{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e2a8e0ee",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from tifffile import imwrite, TiffFile\n",
    "from pathlib import Path\n",
    "from scipy.signal import convolve2d as conv2\n",
    "from scipy.ndimage import zoom\n",
    "from skimage import color, data, restoration\n",
    "from PIL import Image\n",
    "from datetime import datetime\n",
    "import napari\n",
    "\n",
    "\n",
    "# ---------------------\n",
    "# Import 60X 3D data \n",
    "# ---------------------\n",
    "# astro = color.rgb2gray(data.astronaut())\n",
    "path_movie = Path(\n",
    "# rf'/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/RawData/251128/CFL2510A005/Run10/Run10_MMStack_Pos0.ome.tif'\n",
    "rf'/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/RawData/260301/CFL2601A045/Run04/Run04_MMStack_Pos0.ome.tif'\n",
    ")\n",
    "\n",
    "with TiffFile(path_movie) as tif:\n",
    "    stack = tif.asarray()\n",
    "print(stack.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c445e7db",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "andy-image-processing",
   "language": "python",
   "name": "andy-image-processing"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
