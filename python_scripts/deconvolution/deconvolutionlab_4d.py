import subprocess
from pathlib import Path
import tempfile

import numpy as np
from tifffile import imread, imwrite, TiffFile
import xml.etree.ElementTree as ET
import tifffile

# -----------------------
# FUNCTIONS
# -----------------------

def run_dl2_rl(image_path, psf_path, out_dir, out_name, n_iter):
    """
    Run DeconvolutionLab2 Richardson-Lucy via CLI on a single 3D image/PSF pair.
    """

    cmd = [
        java_path, "-jar", str(jar_path), "Run",
        "-image", "file", str(image_path),
        "-psf", "file", str(psf_path),
        "-algorithm", "RL", str(n_iter),
        "-out", "stack", out_name,
        "-path", str(out_dir),
    ]

    print("Running:", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("DL2 stdout:", result.stdout)
        print("DL2 stderr:", result.stderr)
        raise RuntimeError(
            f"DeconvolutionLab2 failed on {image_path}"
        )

    return Path(out_dir) / f"{out_name}.tif"


def read_video_stack(video_path):
    """
    Read TIFF and return data + axes order.
    Example axes: 'TZYX', 'ZYX', etc.
    """

    with TiffFile(video_path) as tif:
        series = tif.series[0]
        axes = series.axes
        data = series.asarray()

    return data, axes


def deconvolve_4d(
    image_path,
    psf_path,
    out_dir,
    n_iter,
):
    """
    Deconvolve every time point of a 4D TZYX TIFF independently.

    Returns
    -------
    output_path : Path
        Path to the final 4D deconvolved TIFF.
    """

    image_path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # READ INPUT
    # -----------------------

    data, axes = read_video_stack(image_path)

    print(f"Input: {image_path.name}")
    print(f"Shape: {data.shape}")
    print(f"Axes:  {axes}")

    # -----------------------
    # HANDLE 3D INPUT
    # -----------------------

    if axes in ("ZYX", "QYX"):
        print("Input is already 3D. Deconvolving directly.")

        output_name = f"{image_path.stem}_DL2_RL{n_iter}"

        return run_dl2_rl(
            image_path=image_path,
            psf_path=psf_path,
            out_dir=out_dir,
            out_name=output_name,
            n_iter=n_iter,
        )

    # -----------------------
    # HANDLE 4D INPUT
    # -----------------------

    if axes != "TZYX":
        raise ValueError(
            f"Expected axes 'TZYX' or 'ZYX', but got '{axes}'."
        )

    n_time = data.shape[0]

    print(f"Found {n_time} time points.")
    print(f"Each frame: {data.shape[1:]}")

    # Temporary directory for individual 3D stacks
    with tempfile.TemporaryDirectory() as tmp_dir:

        tmp_dir = Path(tmp_dir)

        deconvolved_frames = []

        for t in range(n_time):

            print(
                f"\n--- Time point {t + 1}/{n_time} ---"
            )

            # Extract one 3D ZYX stack
            stack_3d = data[t]

            input_stack_path = (
                tmp_dir / f"{image_path.stem}_t{t:04d}.tif"
            )

            output_name = (
                f"{image_path.stem}_t{t:04d}_DL2_RL{n_iter}"
            )

            # Write temporary 3D stack
            imwrite(
                input_stack_path,
                stack_3d,
                metadata={"axes": "ZYX"},
            )

            # Deconvolve this time point
            output_stack_path = run_dl2_rl(
                image_path=input_stack_path,
                psf_path=psf_path,
                out_dir=tmp_dir,
                out_name=output_name,
                n_iter=n_iter,
            )

            # Read deconvolved stack
            deconvolved_stack = imread(output_stack_path)

            print(
                f"Deconvolved shape: {deconvolved_stack.shape}"
            )

            deconvolved_frames.append(deconvolved_stack)

        # -----------------------
        # REASSEMBLE 4D DATA
        # -----------------------

        deconvolved_data = np.stack(
            deconvolved_frames,
            axis=0,
        )

    print(
        f"\nFinal deconvolved shape: "
        f"{deconvolved_data.shape}"
    )

    # -----------------------
    # SAVE FINAL 4D TIFF
    # -----------------------

    output_path = (
        out_dir
        / f"{image_path.stem}_DL2_RL{n_iter}.tif"
    )

    imwrite(
        output_path,
        deconvolved_data,
        metadata={"axes": "TZYX"},
    )

    print(f"Saved: {output_path}")

    return output_path

def load_ome_tiff_as_tczyx(tif_path):
    """
    Load an OME-TIFF and reconstruct its dimensions as (T, C, Z, Y, X).
    """

    with TiffFile(tif_path) as tif:

        series = tif.series[0]
        raw = series.asarray()

        # Read dimensions from OME metadata
        attribs = extract_metadata_attribs(tif_path)

        nt = int(attribs["SizeT"])
        nc = int(attribs["SizeC"])
        nz = int(attribs["SizeZ"])
        ny = int(attribs["SizeY"])
        nx = int(attribs["SizeX"])

        expected_shape = (nt, nc, nz, ny, nx)

        if raw.size != np.prod(expected_shape):
            raise ValueError(
                f"Cannot reshape {raw.shape} into {expected_shape}. "
                f"Raw elements: {raw.size}, "
                f"expected: {np.prod(expected_shape)}"
            )

        stack = raw.reshape(expected_shape)

    return stack

def extract_metadata_attribs(tif_file: Path) -> dict:
    """Extract metadata attributes from a tif file."""  
    with tifffile.TiffFile(tif_file) as tif:
        ome_xml = tif.ome_metadata
        root = ET.fromstring(ome_xml)
        ns = {'ome': root.tag.split('}')[0].strip('{')}
        pixels = root.find('.//ome:Pixels', ns)
        attribs = pixels.attrib
    return attribs

# -----------------------
# SETUP PARAMS
# -----------------------

java_path = "java"

jar_path = Path(
    "/Applications/Fiji/plugins/DeconvolutionLab_2.jar"
)

psf_path = (
    "/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/PSF/Nikon_CFI_PlanApo_VC_60X_WI/psf_3d_resampled_norm.tif"
)

n = 10

out_dir = (
    "/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/PSF/PSF_data/deconvolution_test_data/python_output"
)

data = [
    # "/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/Analysis/260702/CFL2605A123/Run04/Run04_singleframe.tif",
    '/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/RawData/260702/CFL2605A123/Run04_150430/Stack_Run04_150430/Stack_Run04_150430_MMStack_Pos0.ome.tif'
]

# -----------------------
# TEST TIFF LOADING
# -----------------------

for img in data:
    stack = load_ome_tiff_as_tczyx(Path(img))

    print(f"\nFile: {Path(img).name}")
    print(f"Loaded shape: {stack.shape}")

with TiffFile(Path(data[0])) as tif:

    ome = tif.ome_metadata

    start = ome.find("<Pixels")

    end = ome.find("</Pixels>") + len("</Pixels>")

    print(ome[start:end])

# -----------------------
# RUN
# -----------------------

# for img in data:

#     output_path = deconvolve_4d(
#         image_path=img,
#         psf_path=psf_path,
#         out_dir=out_dir,
#         n_iter=n,
#     )

#     print(f"Saved: {output_path}")