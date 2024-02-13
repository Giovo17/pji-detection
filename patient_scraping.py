import os
import numpy as np
import pydicom
from PIL import Image
from typing import Tuple



from remove_folders import remove_folders


# Paths
data_path = os.getcwd() + "/data"
export_data_path = os.getcwd() + "/data/png_export"
export_resized_path = data_path + "/png_resized"
#export_rescaled_path = export_resized_path + "/rescaled"
#export_cropped_path = export_resized_path + "/cropped"
infected_path = data_path + "/Pazienti_Settici"
aseptic_path = data_path + "/Pazienti_Asettici"
infected_export_path = export_data_path + "/Pazienti_Settici"
aseptic_export_path = export_data_path + "/Pazienti_Asettici"


def convert_dcm_png(path, desired_size: Tuple[int] = None):
    """Convert an ..."""
    
    im = pydicom.dcmread(path)
    im = im.pixel_array.astype(float)

    rescaled_image = (np.maximum(im,0)/im.max())*255 # float pixels
    final_image = np.uint8(rescaled_image) # integers pixels
    final_image = Image.fromarray(final_image)

    print(type(final_image))

    #final_image.crop()

    if desired_size != None:
        final_image = final_image.resize(desired_size)

    return final_image



def preprocess_patients(import_path: str, export_path: str, desired_size, rescaled: bool = False, cropped: bool = False):
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
                            #os.makedirs(export_resized_path + '/' + 'cropped' +'_'+ str(desired_size[0])+'x'+str(desired_size[0]), exist_ok=True)
                            #image_converted = convert_dcm_png(el_path, desired_size)
                            #image_converted.save(export_resized_path + '/' + 'cropped' +'_'+ str(desired_size[0])+'x'+str(desired_size[0]) + "/" + folder + "/" + int_f+ "/" + el +'.png')
                            
                        elif rescaled:
                            os.makedirs(export_resized_path + '/rescaled_'+ str(desired_size[0])+'x'+str(desired_size[0]), exist_ok=True)
                            image_converted = convert_dcm_png(el_path, desired_size)
                            image_converted.save(export_resized_path + '/rescaled_'+ str(desired_size[0])+'x'+str(desired_size[0]) + "/" + folder + "/" + int_f+ "/" + el +'.png')


                    else:
                        print("No image data in DICOM file")
            
                except Exception as e:
                    print("Error reading DICOM file:", str(e))



def resize_images(path: str, dimension: Tuple[int], action: str):
    os.makedirs(export_resized_path + '/' + action +'_'+ str(dimension[0])+'x'+str(dimension[0]), exist_ok=True)
    preprocess_patients(aseptic_export_path, export_resized_path)


    print("aa")




def main():
    #preprocess_patients(aseptic_path, aseptic_export_path)
    preprocess_patients(aseptic_path, aseptic_export_path, desired_size = (224,224), rescaled = True)
    print("Aseptic end\n")

    #preprocess_patients(infected_path, infected_export_path)
    preprocess_patients(infected_path, infected_export_path, desired_size = (224,224), rescaled = True)

    print("Infected end\n")


    remove_folders(export_data_path)


if __name__ == "__main__":
    main()

