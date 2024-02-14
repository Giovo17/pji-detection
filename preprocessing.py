import os
import numpy as np
import pydicom
from PIL import Image
from PIL import UnidentifiedImageError
from typing import Tuple

from utils.remove_folders import remove_empty_folders


# Paths
data_path = os.path.join(os.getcwd(), "data")
dicom_data_path = os.path.join(data_path, "dicom_data")
png_export_data_path = os.path.join(data_path, "png_export")
png_resized_data_path = os.path.join(data_path, "png_resized")


#export_rescaled_path = export_resized_path + "/rescaled"
#export_cropped_path = export_resized_path + "/cropped"


def convert_dcm_png(path, desired_size: Tuple[int] = None):
    """Convert a dicom image in png."""
    
    im = pydicom.dcmread(path)
    im = im.pixel_array.astype(float)

    rescaled_image = (np.maximum(im,0)/im.max())*255 # float pixels
    final_image = np.uint8(rescaled_image) # integers pixels
    final_image = Image.fromarray(final_image)

    return final_image


def preprocess_patients_png(import_path: str, export_path: str):
    """Convert all dicom images into png by recreating the input folder structure."""

    os.makedirs(export_path, exist_ok=True)

    import_dir = import_path.split("/")[-1]
    print(f"IMPORT: {import_dir}")
    export_dir = export_path.split("/")[-1]
    print(f"EXPORT: {export_dir}")


    # Cycle through all subfolder and recreate them into export_path
    for root, dirs, files in os.walk(import_path, topdown=False):
        for name in dirs:
            os.makedirs(os.path.join(root, name).replace(import_dir, export_dir), exist_ok=True)

    # Cycle through all files and convert them to png and save to output directory
    for root, dirs, files in os.walk(import_path, topdown=False):
        for name in files:            
            file_path = os.path.join(root, name)
            output_name = file_path.split("/")[-4] + "_" + name
            output_file_path = os.path.join(root, output_name).replace(import_dir, export_dir).split(".")[0] + ".png"
            #print(output_file_path)
            try:
                dicom_data = pydicom.dcmread(file_path)
                if 'PixelData' in dicom_data: # Check if the DICOM data contains pixel data
                    pixel_array = dicom_data.pixel_array
                    height, width = pixel_array.shape
                    
                    if (height, width) == (512,512): # Filter only axial tomographies
                        image_converted = convert_dcm_png(file_path)
                        image_converted.save(output_file_path)

                else:
                    print("No image data in DICOM file")
        
            except Exception as e:
                print(f"Error reading DICOM file: {e}")


    # Remove empty folders
    remove_empty_folders(export_path)


def resize_patient_png(import_path: str, export_path: str, desired_size: Tuple[int], technique: str = "rescale") -> None:
    """Convert all dicom images into png by recreating the input folder structure."""

    os.makedirs(export_path, exist_ok=True)

    # Create a subfolder devoted to resize technique and the desided size
    export_path = os.path.join(export_path, technique + "_" + str(desired_size[0])+'x'+str(desired_size[1])) # Adjust the export path
    os.makedirs(export_path, exist_ok=True)


    import_dir = import_path.split("/")[-1]
    print(f"IMPORT: {import_dir}")
    export_dir = export_path.split("/")[-2] + "/" + export_path.split("/")[-1]
    print(f"EXPORT: {export_dir}")

    # Cycle through all subfolder and recreate them into export_path
    for root, dirs, files in os.walk(import_path, topdown=False):
        for name in dirs:
            os.makedirs(os.path.join(root, name).replace(import_dir, export_dir), exist_ok=True)

    # Cycle through all files and convert them to png and save to output directory
    for root, dirs, files in os.walk(import_path, topdown=False):
        for name in files:            
            file_path = os.path.join(root, name)
            try:
                image = Image.open(file_path)
            except UnidentifiedImageError as e:
                print(f"cannot identify image file {file_path}")
                continue

            resized_image = image.resize(desired_size)

            output_file_path = file_path.replace(import_dir, export_dir).split(".")[0] + ".png"
            resized_image.save(output_file_path)


    # Remove empty folders
    remove_empty_folders(export_path)


def main():
    
    preprocess_patients_png(dicom_data_path, png_export_data_path)

    resize_patient_png(png_export_data_path, png_resized_data_path, (227,227))


if __name__ == "__main__":
    main()

