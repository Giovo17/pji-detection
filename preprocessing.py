import os
import numpy as np
import pydicom
from PIL import Image
from PIL import UnidentifiedImageError
from typing import Tuple

from utils.remove_folders import remove_empty_folders
from single_image_preprocessing import convert_dcm2matrix, hounsfield_hp_detection, bone_extraction, image_equalization


# Paths
data_path = os.path.join(os.getcwd(), "data")
dicom_data_path = os.path.join(data_path, "dicom_data")
png_export_data_path = os.path.join(data_path, "png_export")
png_resized_data_path = os.path.join(data_path, "png_resized")


#export_rescaled_path = export_resized_path + "/rescaled"
#export_cropped_path = export_resized_path + "/cropped"




def preprocess_patients(import_path: str, export_path: str):
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

                # Check if the DICOM data contains desired metadata
                if 'PixelData' in dicom_data and 'RescaleSlope' in dicom_data and 'RescaleIntercept' in dicom_data:
                    pixel_array = dicom_data.pixel_array
                    height, width = pixel_array.shape
                    
                    if (height, width) == (512,512): # Filter only axial tomographies
                        #image_converted = convert_dcm_png(file_path) # Deprecated
                        #image_converted.save(output_file_path) # Deprecated
                        image_matrix = convert_dcm2matrix(file_path)

                        if hounsfield_hp_detection(image_matrix, 
                                                   float(dicom_data.RescaleSlope), 
                                                   float(dicom_data.RescaleIntercept), 
                                                   threshold=3000):
                            image_bone = bone_extraction(image_matrix, 
                                                         float(dicom_data.RescaleSlope), 
                                                         float(dicom_data.RescaleIntercept), 
                                                         metal_threshold=3000,
                                                         bone_threshold=1000, 
                                                         output_dim=(188,188))
                            
                            image_bone_equalized = image_equalization(image_bone)
                            
                            image_bone_equalized.save(output_file_path)


                else:
                    print("No image data in DICOM file")
        
            except Exception as e:
                print(f"Error reading DICOM file: {e}")


    # Remove empty folders
    remove_empty_folders(export_path)


def resize_patients(import_path: str, export_path: str, desired_size: Tuple[int], technique: str = "rescale") -> None:
    """Convert all dicom images into png by recreating the input folder structure."""

    os.makedirs(export_path, exist_ok=True)

    # Create a subfolder devoted to resize technique and the desided size
    export_path = os.path.join(export_path, technique + "_" + str(desired_size[0])+'x'+str(desired_size[1])) # Adjust the export path
    os.makedirs(export_path, exist_ok=True)

    # Create infected and normal patients folders
    os.makedirs(os.path.join(export_path, "Pazienti_Asettici"), exist_ok=True)
    os.makedirs(os.path.join(export_path, "Pazienti_Settici"), exist_ok=True)
    

    import_dir = import_path.split("/")[-1]
    print(f"IMPORT: {import_dir}")
    export_dir = export_path.split("/")[-2] + "/" + export_path.split("/")[-1]
    print(f"EXPORT: {export_dir}")


    # Cycle through all files and convert them to png and save to output directory
    for root, dirs, files in os.walk(import_path, topdown=False):
        for name in files:            
            file_path = os.path.join(root, name)
            try:
                image = Image.open(file_path)
            except UnidentifiedImageError as e:
                print(f"cannot identify image file {file_path}")
                print(e)
                continue

            resized_image = image.resize(desired_size)

            resized_image = resized_image.convert("RGB") # Convert to RGB cause almost every CNN take as input 3-channels images
            
            # Output filepath
            splitted_file_path = file_path.split("/")
            img_id = splitted_file_path[-1].split("Patient")[-1][2:]
            patient = splitted_file_path[-1].split("Patient")[-1][0:2]
            splitted_file_path.pop(-1)
            while(len(splitted_file_path) > 0):
                if "Patient" in splitted_file_path[-1]:
                    splitted_file_path.pop()
                    break
                img_id = splitted_file_path[-1] + "_" + img_id
                splitted_file_path.pop()
            
            base_output_path = "/".join(splitted_file_path).replace(import_dir, export_dir)
            output_file_path = os.path.join(base_output_path, "Patient" + patient + "_" + img_id)
                
            resized_image.save(output_file_path)


    # Remove empty folders
    remove_empty_folders(export_path)


def main():
    
    print("Init preprocessing")

    preprocess_patients(dicom_data_path, png_export_data_path)

    resize_patients(png_export_data_path, png_resized_data_path, (227,227))
    resize_patients(png_export_data_path, png_resized_data_path, (224,224))
    

if __name__ == "__main__":
    main()

