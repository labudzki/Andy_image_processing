import subprocess
from pathlib import Path
import numpy as np
from tifffile import imread, imwrite, TiffFile

java_path = "/Applications/Fiji/java/macos-arm64/zulu21.42.19-ca-jdk21.0.7-macosx_aarch64/zulu-21.jdk/Contents/Home/bin/java"
java_path = "java"
jar_path = Path("/Applications/Fiji/plugins/DeconvolutionLab_2.jar")

def run_dl2_rl(image_path, psf_path, out_dir, out_name, n_iter):
    """Run DeconvolutionLab2 Richardson-Lucy via CLI on a single image/PSF pair."""
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
        raise RuntimeError(f"DeconvolutionLab2 failed on {image_path}")

    return Path(out_dir) / f"{out_name}.tif"

# example: loop over a batch of stacks
data = [
    "/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/Analysis/260702/CFL2605A123/Run04/Run04_singleframe.tif",
    # '/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/5. Lipids and Organelles imaging/RawData/260702/CFL2605A123/Run04_150430/Stack_Run04_150430/Stack_Run04_150430_MMStack_Pos0.ome.tif'
    # "/path/to/stack2.tif"
]

psf_path = '/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/PSF/Nikon_CFI_PlanApo_VC_60X_WI/psf_3d_resampled_norm.tif'

n = 10

for img in data:
    out_path = run_dl2_rl(img, psf_path, out_dir="/Users/andrealabudzki/Library/CloudStorage/Dropbox-AMOLF-SHIMIZU/DATA/Ach_data/x. SetUp Charac/PSF/PSF_data/deconvolution_test_data/python_output", out_name=Path(img).stem + "_DL2_RL50", n_iter = n)
    print(f"saved: {out_path}")

