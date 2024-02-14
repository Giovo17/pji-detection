import os
import numpy as np
import pydicom
from PIL import Image
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
    """Convert an ..."""
    
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
            #print(file_path)
            try:
                dicom_data = pydicom.dcmread(file_path)
                if 'PixelData' in dicom_data: # Check if the DICOM data contains pixel data
                    pixel_array = dicom_data.pixel_array
                    height, width = pixel_array.shape
                    
                    if (height, width) == (512,512): # Filter only axial tomographies
                        image_converted = convert_dcm_png(file_path)
                        output_file_path = file_path.replace(import_dir, export_dir).split(".")[0] + ".png"
                        image_converted.save(output_file_path)

                else:
                    print("No image data in DICOM file")
        
            except Exception as e:
                print(f"Error reading DICOM file: {e}")


def resized_patient_png(import_path: str, export_path: str, desired_size:Tuple[int]):
    """Convert all dicom images into png by recreating the input folder structure."""

    os.makedirs(export_path, exist_ok=True)

    import_dir = import_path.split("/")[-1]
    print(f"IMPORT: {import_dir}")
    export_dir = export_path.split("/")[-1]
    print(f"EXPORT: {export_dir}")

    # Cycle through all subfolder and recreate them into export_path
    for root, dirs, files in os.walk(import_path, topdown=False):
        for name in dirs:
            os.makedirs(os.path.join(root, name).replace(import_dir, export_dir + '/' + export_dir + str(desired_size)[0]+'x'+str(desired_size)[0]), exist_ok=True)

    # Cycle through all files and convert them to png and save to output directory
    for root, dirs, files in os.walk(import_path, topdown=False):
        for name in files:            
            file_path = os.path.join(root, name)
            image = Image.open(file_path)
            resized_image = image.resize(desired_size)

            output_file_path = file_path.replace(import_dir, export_dir + '/' + export_dir + str(desired_size)[0]+'x'+str(desired_size)[0]).split(".")[0] + ".png"
            resized_image.save(output_file_path)













def preprocess_patients_old(import_path: str, export_path: str, desired_size, rescaled: bool = False, cropped: bool = False):
    os.makedirs(export_path, exist_ok=True)
    
    for folder in os.listdir(import_path):
        print("Patient: " + folder)
        if folder == '.DS_Store':
            continue
        if rescaled:
            os.makedirs(export_resized_path + '/rescaled_'+ str(desired_size[0])+'x'+str(desired_size[0]), exist_ok=True)
            if folder in os.listdir(export_resized_path + '/rescaled_'+ str(desired_size[0])+'x'+str(desired_size[0])):
                continue
        elif cropped:
            os.makedirs(export_resized_path + '/cropped_'+ str(desired_size[0])+'x'+str(desired_size[0]), exist_ok=True)
            if folder in os.listdir(export_resized_path + '/cropped_'+ str(desired_size[0])+'x'+str(desired_size[0])):
                continue
        elif folder in os.listdir(export_path):
            continue

        folder_internal = [f for f in os.listdir(import_path + "/" + folder) if f not in ['DICOMDIR', 'LOCKFILE', 'VERSION','.DS_Store']][0]
        
        internal_folders = [f for f in os.listdir(import_path + "/" + folder + "/" + folder_internal) if f not in ['DICOMDIR', 'LOCKFILE', 'VERSION','.DS_Store']]
        
        
        for int_f in internal_folders:
            os.makedirs(export_path + "/" + folder + "/" + int_f + "/", exist_ok=True)  # Create directory if it doesn't exist

            files = os.listdir(import_path + "/" + folder + "/" + folder_internal + "/" + int_f)
            for el in files:
                el_path = import_path + "/" + folder + "/" + folder_internal + "/" + int_f + "/" + el
                #print("file is :" + el_path)
                try:
                    dicom_data = pydicom.dcmread(el_path)
                    # Check if the DICOM data contains pixel data
                    if 'PixelData' in dicom_data:
                        # Get the pixel array
                        pixel_array = dicom_data.pixel_array
                        # Get the dimensions of the pixel array (image)
                        height, width = pixel_array.shape
                        #print("Image Dimensions (Height x Width):", height, "x", width)
                        if (height, width) == (512,512) and cropped == False and rescaled == False:
                            image_converted = convert_dcm_png(el_path)
                            image_converted.save(export_path + "/" + folder + "/" + int_f+ "/" + el +'.png')
                        
                        elif cropped:
                            continue
                            # (TO-DO) To implement
                            #os.makedirs(export_resized_path + '/' + 'cropped' +'_'+ str(desired_size[0])+'x'+str(desired_size[0])+ "/" + folder + "/" + int_f, exist_ok=True)
                            #image_converted = convert_dcm_png(el_path, desired_size)
                            #image_converted.save(export_resized_path + '/' + 'cropped' +'_'+ str(desired_size[0])+'x'+str(desired_size[0]) + "/" + folder + "/" + int_f+ "/" + el +'.png')
                            
                        elif rescaled:
                            os.makedirs(export_resized_path + '/rescaled_'+ str(desired_size[0])+'x'+str(desired_size[0])+ "/" + folder + "/" + int_f, exist_ok=True)
                            image_converted = convert_dcm_png(el_path, desired_size)
                            image_converted.save(export_resized_path + '/rescaled_'+ str(desired_size[0])+'x'+str(desired_size[0]) + "/" + folder + "/" + int_f+ "/" + el +'.png')


                    else:
                        print("No image data in DICOM file")
            
                except Exception as e:
                    print("Error reading DICOM file:", str(e))
                    raise e



def resize_images(path: str, dimension: Tuple[int], action: str):
    os.makedirs(export_resized_path + '/' + action +'_'+ str(dimension[0])+'x'+str(dimension[0]), exist_ok=True)
    preprocess_patients(aseptic_export_path, export_resized_path)


    #final_image.crop()

    if desired_size != None:
        final_image = final_image.resize(desired_size)

    print("aa")




def main():
    
    preprocess_patients_png(dicom_data_path, png_export_data_path)

    #preprocess_patients_resize(png_export_data_path, png_resized_data_path)

    #remove_empty_folders(png_export_data_path)


if __name__ == "__main__":
    main()

