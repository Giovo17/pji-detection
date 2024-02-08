import os
import pydicom
import numpy as np
from PIL import Image


# Paths
data_path = os.getcwd() + "/data"
infected_path = data_path + "/Pazienti_Settici"
aseptic_path = data_path + "/Pazienti_Asettici"



def get_names(path):
    names = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext in ['.dcm']:
                names.append(filename)
    
    return names

def convert_dcm_jpg(name):
    
    im = pydicom.dcmread('Database/'+name)

    im = im.pixel_array.astype(float)

    rescaled_image = (np.maximum(im,0)/im.max())*255 # float pixels
    final_image = np.uint8(rescaled_image) # integers pixels

    final_image = Image.fromarray(final_image)

    return final_image



for folder in os.listdir(infected_path):
    print("Patient: " + folder)
    folder_internal = [f for f in os.listdir(infected_path + "/" + folder) if f not in ['DICOMDIR', 'LOCKFILE', 'VERSION']][0]
    
    internal_folders = os.listdir(infected_path + "/" + folder + "/" + folder_internal)
    
    # (TO-DO) Scegliere la cartella che ha più elementi al suo interno
    for int_f in internal_folders:
        print(int_f)
    

    # (TO-DO) Convertire le immagini dentro la cartella di riferimento a PNG
        
    

    print("---------------------------\n")
    break
    











#names = get_names('Database')
#for name in names:
#    image = convert_dcm_jpg(name)
#    image.save(name+'.png')





