General instructions for deconvolution
- open the image of beads in napari
- create a new layer for drawing shapes. draw boxes around the beads that you would like to include in your PSF measurement 
- export the rectange layer as csv
- import to PSF_measurement_csv.py to csv_path
- import the image of the beads as well, to image_path
- PSF_measurement_csv.py will take the intensity profiles of the beads located in the rectangles, align them and average them to give an averaged PSF
- the code in LocoMyco assumes the PSF is ready.


-------------------------
2D
-------------------------
psf_measurement_csv.py
This script measures the PSF from a pre-existing CSV file. See General instructions above for obtaining csv file.

    Loads a microscopy image of fluorescent nanobeads 
    Reads a CSV of manually annotated bead coordinates
    For each bead, crops a small window around its intensity peak and normalizes it
    Aligns all bead crops to a common reference using cross-correlation
    Averages them together to produce a clean, noise-reduced PSF
    Saves the result as .npy, .csv, and .png files
-------------------------
plot_deconvolution.py:
    This script applies Richardson-Lucy deconvolution to sharpen a fluorescence microscopy image.

    Loads a single time frame from a multi-frame .ome.tif microscopy stack
    Loads an experimentally measured PSF (point spread function) for either a 20x or 60x objective
    Runs RL deconvolution at 10, 20, and 30 iterations to compare sharpening strength
    Displays the original vs the 10-iteration result
    Saves the results as .png (all iterations) and .tif (30 iterations only)

-------------------------
3D
-------------------------
psf_measurement_csv_3d.py
This script measures the PSF from a pre-existing CSV file. See General instructions above for obtaining csv file.

OLD

-------------------------
psf_measurement.py:

Interactive napari-based tool for measuring the PSF of a single bead from a microscopy image. 
Draw a rectangle ROI around a bead in the viewer, and the script automatically crops it, 
fits a 1D Gaussian to the summed intensity profile, and prints the FWHM in pixels and µm. 
Theoretical resolution limits (Rayleigh, Sparrow, Abbe) are printed at startup for comparison. 
Intended as a quick diagnostic tool — for multi-bead averaging and PSF export, see the other PSF scripts.

-------------------------
psf_measurement_extracting_averaging.py DOESNT WORK VERY WELL

This is an improved, interactive version of psf_measurement.py. Like the original, you manually select beads in napari, but it's much more fully featured:

Per-bead workflow: draw a rectangle around each bead → it automatically finds the peak, extracts a fixed-size window, and marks the center with a yellow dot
Delete tools: press D to delete a selected bead, or Backspace to remove the last one — useful for rejecting bad/out-of-focus beads
Press A to average: normalizes and subpixel-aligns all collected beads, then computes the averaged 2D PSF, plots X/Y profiles with Gaussian fits, prints FWHM in pixels and µm, and saves psf_average.npy
