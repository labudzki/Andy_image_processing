import tifffile
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import warnings
import os
import contextlib

def extract_fps(tif_file: Path) -> float:
    """Extract mean FPS from a tif file."""  
    planes = extract_planes(tif_file)
    delta_ts = []
    for plane in planes:
        dt = plane.attrib.get('DeltaT')
        if dt is not None:
            delta_ts.append(float(dt))
    delta_ts = np.array(delta_ts)
    frame_intervals = np.diff(delta_ts)
    if len(frame_intervals)==0:
        return 1.0
    mean_interval = np.mean(frame_intervals)
    fps = 1e3/ mean_interval 
    return fps  # mean FPS

def extract_metadata_attribs(tif_file: Path) -> dict:
    """Extract metadata attributes from a tif file."""  
    with tifffile.TiffFile(tif_file) as tif:
        ome_xml = tif.ome_metadata
        root = ET.fromstring(ome_xml)
        ns = {'ome': root.tag.split('}')[0].strip('{')}
        pixels = root.find('.//ome:Pixels', ns)
        attribs = pixels.attrib
    return attribs

def extract_planes(tif_file: Path):
    """Extract planes attributes from a tif file."""  
    with tifffile.TiffFile(tif_file) as tif:
        ome_xml = tif.ome_metadata
        root = ET.fromstring(ome_xml)
        ns = {'ome': root.tag.split('}')[0].strip('{')}
        planes = root.findall('.//ome:Plane', ns)
    return planes

def load_tiff_stack(tif_path: Path, squeeze=True) -> np.ndarray:
    """Load a TIFF stack from the given path. Reshape to (T, Z, C, Y, X)."""
    warnings.simplefilter("error")
    with tifffile.TiffFile(tif_path) as tif:
        with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f):
            expected_nt=tif.series[0].shape[0]
        attribs=extract_metadata_attribs(tif_path)
        nt=int(attribs.get('SizeT'))
        nz=int(attribs.get('SizeZ'))
        nc=int(attribs.get('SizeC'))
        height=int(attribs.get('SizeY'))
        witdh=int(attribs.get('SizeX'))
        if nt==expected_nt:
            stack = tif.asarray().reshape((int(nt),int(nc),int(nz),int(height),int(witdh)))
        else:
            planes=extract_planes(tif_path)
            last_attrib=planes[-1].attrib
            last_c=int(last_attrib.get('TheC'))
            last_z=int(last_attrib.get('TheZ'))
            if last_c==nc-1 and last_z==nz-1:
                stack=np.zeros((int(nt),int(nz),int(nc),int(height),int(witdh)),dtype=np.uint16)
            else:
                nt=nt-1
                stack=np.zeros((int(nt),int(nz),int(nc),int(height),int(witdh)),dtype=np.uint16)
            for i,page in enumerate(tif.pages):
                plane=planes[i]
                t=int(plane.attrib.get('TheT'))
                z=int(plane.attrib.get('TheZ'))
                c=int(plane.attrib.get('TheC'))
                stack[t,z,c,:,:]=page.asarray()
        if squeeze:
            stack = np.squeeze(stack)
    return stack

def stack4D_type(tif_path : Path) -> str:
    attribs=extract_metadata_attribs(tif_path)
    nt=int(attribs.get('SizeT'))
    nz=int(attribs.get('SizeZ'))
    if nt==1 and nz==1:
        return "single image"
    elif nt>1 and nz==1:
        return "2D movie"
    elif nt>1 and nz>1:
        return "3D movie"
    elif nt==1 and nz>1:
        return "3D stack"







