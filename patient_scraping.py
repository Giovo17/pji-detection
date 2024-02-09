import os
import pydicom
import numpy as np
from PIL import Image
from pathlib import Path

# Paths
data_path = os.getcwd() + "/data"
infected_path = data_path + "/Pazienti_Settici"
aseptic_path = data_path + "/Pazienti_Asettici"
infected_output_image_path = data_path + "/png_export/Pazienti_Settici"
aseptic_output_image_path = data_path + "/png_export/Pazienti_Asettici"



def get_names(path):
    names = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext in ['.dcm']:
                names.append(filename)
    
    return names

def convert_dcm_jpg(path):
    
    im = pydicom.dcmread(path)

    im = im.pixel_array.astype(float)

    rescaled_image = (np.maximum(im,0)/im.max())*255 # float pixels
    final_image = np.uint8(rescaled_image) # integers pixels

    final_image = Image.fromarray(final_image)

    return final_image



for folder in os.listdir(infected_path):
    print("Patient: " + folder)
    folder_internal = [f for f in os.listdir(infected_path + "/" + folder) if f not in ['DICOMDIR', 'LOCKFILE', 'VERSION']][0]
    
    internal_folders = [f for f in os.listdir(infected_path + "/" + folder + "/" + folder_internal) if f not in ['DICOMDIR', 'LOCKFILE', 'VERSION']]
    print(internal_folders)
    
    # (TO-DO) Scegliere la cartella che ha più elementi al suo interno
    
    for int_f in internal_folders:
        print(int_f)
        files = os.listdir(infected_path + "/" + folder + "/" + folder_internal + "/" + int_f)
        for el in files:
            el_path = infected_path + "/" + folder + "/" + folder_internal + "/" + int_f + "/" + el
            print("file is :" + el_path)
            try:
                dicom_data = pydicom.dcmread(el_path)
                # Check if the DICOM data contains pixel data
                if 'PixelData' in dicom_data:
                    # Get the pixel array
                    pixel_array = dicom_data.pixel_array
                    # Get the dimensions of the pixel array (image)
                    height, width = pixel_array.shape
                    print("Image Dimensions (Height x Width):", height, "x", width)
                    if (height, width) == (512,512):
                        image_converted = convert_dcm_jpg(el_path)
                        image_converted.save(infected_output_image_path +"/" + el +'.png')
                
                else:
                    print("No image data in DICOM file")
        
            except Exception as e:
                print("Error reading DICOM file:", str(e))
    break

    # (TO-DO) Convertire le immagini dentro la cartella di riferimento a PNG
        
    

    print("---------------------------\n")
    break
    











#names = get_names('Database')
#for name in names:
#    image = convert_dcm_jpg(name)
#    image.save(name+'.png')





