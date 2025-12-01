from email.mime import image
import napari
from magicgui import magicgui
import numpy as np

dummy_stack = np.random.rand(10, 50, 50)  # simple 3D array

viewer = napari.Viewer()
image_layer = viewer.add_image(dummy_stack, name='dummy')

# Add a Shapes layer for ROI selection
roi_layer = viewer.add_shapes(name="ROI", shape_type="rectangle")

# Button to enable ROI selection
@magicgui(call_button="Enable ROI Selection")
def enable_roi():
    roi_layer.mode = 'add_rectangle'  # Switch to rectangle drawing mode
    print("Draw a rectangle on the image to select ROI.")

# # Define crop function
# @magicgui(call_button="Crop ROI")
# def crop_roi():
#     if len(roi_layer.data) == 0:
#         print("Please draw a rectangle ROI first.")
#         return

#     # Get the first ROI rectangle coordinates
#     rect = roi_layer.data[0]  # [[y0, x0], [y1, x1], [y2, x2], [y3, x3]]
#     y_min, x_min = rect.min(axis=0)
#     y_max, x_max = rect.max(axis=0)

#     # Crop image
#     cropped = image[int(y_min):int(y_max), int(x_min):int(x_max)]
#     viewer.add_image(cropped, name="cropped")


# Add Crop button
@magicgui(call_button="Crop ROI")
def crop_roi():
    # Find the Shapes layer
    shapes_layers = [layer for layer in viewer.layers if layer.__class__.__name__ == "Shapes"]
    if not shapes_layers:
        print("Please create a Shapes layer and draw a rectangle ROI.")
        return

    roi_layer = shapes_layers[0]  # Use the first Shapes layer
    if len(roi_layer.data) == 0:
        print("Please draw a rectangle ROI first.")
        return

    # Get rectangle coordinates
    rect = roi_layer.data[0]  # [[y0,x0],[y1,x1],[y2,x2],[y3,x3]]
    y_min, x_min = rect.min(axis=0)
    y_max, x_max = rect.max(axis=0)

    # Crop the current image layer
    current_image = image_layer.data
    cropped = current_image[int(y_min):int(y_max), int(x_min):int(x_max)]

    # Add cropped image as a new layer
    viewer.add_image(cropped, name="Cropped ROI")


# Add widget to viewer
viewer.window.add_dock_widget(enable_roi, area="right")
viewer.window.add_dock_widget(crop_roi, area="right")


napari.run()