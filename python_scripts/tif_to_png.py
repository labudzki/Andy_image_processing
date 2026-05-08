# convert tif to png 
from PIL import Image
import os   

input_path = r"C:\Users\labudzki\AMOLF-SHIMIZU Dropbox\DATA\NETWORK_VISU\011_CFL2510A\stitch\20251202_1822.tif"
output_path = r"C:\Users\labudzki\AMOLF-SHIMIZU Dropbox\Andrea Labudzki\Posters"

def convert_tif_to_png(input_path, output_path):
    # Open the TIFF image
    with Image.open(input_path) as img:
        # Define the output file path
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_file = os.path.join(output_path, f"{base_name}.png")
        
        # Save the image as PNG
        img.save(output_file, "PNG")
        print(f"Converted {input_path} to {output_file}")   

convert_tif_to_png(input_path, output_path)
