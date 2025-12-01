from genericpath import exists
import os
import shutil
import pandas as pd
from glob import glob

#currently not working because the network visu folders are not synced to my local drive

path_excel = r'C:\Users\labudzki\AMOLF-SHIMIZU Dropbox\Andrea Labudzki\Plates\Andy_PlateTracking.xlsx'
path_folder_image = r'C:\Users\labudzki\AMOLF-SHIMIZU Dropbox\Andrea Labudzki\Plates'
path_visu = r'C:\AMOLF-SHIMIZU Dropbox\DATA\NETWORK_VISU'

# Read excel file
track_table = pd.read_excel(path_excel)

# Clear existing .tif files in the target folder
fileList = glob(os.path.join(path_folder_image, '*.tif'))
# list_imgs = [os.path.basename(f)[:11] for f in fileList]

for f in fileList:
    os.remove(f)

# Get indices of plates with Days after cross >= 0
idx = track_table.index[track_table['Days after cross'] >= 0].tolist()
UI_found = []
UI_notfound = []

num_imgs = 1

for i in idx:
    UI = track_table.at[i, 'UI']
    folder_name = f"{UI[-3:]}_{UI[:8]}"
    # print(f"Processing plate {UI} in folder {folder_name}")

    stitch_path = os.path.join(path_visu, folder_name, 'stitch')
    print(f"Looking for images in: {stitch_path}")

    # print(f"Looking for images in: {stitch_path}")
    # print(f"{path_folder_image}\\")
    # print(f"{UI}")
    # Destination subfolder for this plate
    plate_folder = os.path.join(path_folder_image, UI)
    print(f"Destination folder: {plate_folder}")
    if not os.path.exists(plate_folder):
        os.makedirs(plate_folder)  # Create only if missing
        plate_folder = os.path.join(path_folder_image, UI)
    
    
    
    print(f"Checking path: {stitch_path}")
    print(f"Exists? {os.path.exists(stitch_path)}")
    print(f"Contents: {os.listdir(stitch_path)}")

    fileList = glob(os.path.join(stitch_path, '*.tif'))
    # fileList = sorted(glob(os.path.join(stitch_path, '*.tif')))

    if fileList:
            # Get last num_imgs files
            last_images = fileList[-num_imgs:]

            for img_path in last_images:
                filename_lastpic = os.path.basename(img_path)
                filename_new = f"{track_table.at[i, 'Days after cross']}d_{UI}_{filename_lastpic}.tif"
                shutil.copyfile(img_path, os.path.join(path_folder_image, filename_new))

            UI_found.append(UI)
    else:
        UI_notfound.append(UI)

    # if fileList:
    #     filename_lastpic = os.path.basename(fileList[-1])
    #     filename_new = f"{track_table.at[i, 'Days after cross']}d_{UI}_{filename_lastpic}.tif"

    #     # Ensure destination folder exists
    #     if not os.path.exists(path_folder_image):
    #         os.makedirs(path_folder_image)

    #     shutil.copyfile(os.path.join(stitch_path, filename_lastpic),
    #                     os.path.join(path_folder_image, filename_new))
    #     UI_found.append(UI)
    # else:
    #     UI_notfound.append(UI)

print('Transferred stitch image of plates:')
for u in UI_found:
    print(u)
print('Could not find folders of the plates')
for u in UI_notfound:
    print(u)