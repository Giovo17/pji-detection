import os
import numpy as np
import pydicom
from PIL import Image
from PIL import UnidentifiedImageError
from typing import Tuple
from itertools import groupby

# DEPRECATED
def convert_dcm_png(path, desired_size: Tuple[int] = None):
    """Convert a dicom image in png."""
    
    im = pydicom.dcmread(path)
    im = im.pixel_array.astype(float)

    rescaled_image = (np.maximum(im,0)/im.max())*255 # float pixels
    final_image = np.uint8(rescaled_image) # integers pixels
    final_image = Image.fromarray(final_image)

    return final_image



def convert_dcm2matrix(file_path: str) -> np.ndarray:
    """Convert a dicom image to numpy matrix."""

    im = pydicom.dcmread(file_path)
    im = im.pixel_array.astype(float)

    return im



def hounsfield_hp_detection(image: np.ndarray, slope: float, intercept: float, threshold: int) -> bool:
    """Hounsfield based prothesis detection."""

    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i,j] * slope + abs(intercept) > threshold:
                return True

    return False


def bone_extraction(image: np.ndarray, slope: float, intercept: float, threshold: int, output_dim: Tuple[int]) -> np.ndarray:
    """Bone patch extraction."""

    # Bone coordinates detection
    bone_dict= {}
    bone_coordinates = []
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i,j] * slope + intercept > threshold:
                if i not in bone_dict:
                    bone_dict[i] = [(i,j)]
                else:
                    # Se i è già presente, aggiungi la tupla (i, j) alla lista corrispondente
                    bone_dict[i].append((i,j))
                bone_coordinates.append((i,j))

    print(bone_dict)

    list_temp_mean_i = []
    for k,v in bone_dict.items():
        somma = 0
        for tup in v:
            somma+=tup[1]
        temp_mean = somma/len(v)
        list_temp_mean_i.append(temp_mean)

    mean_for_row = sum(list_temp_mean_i)/len(list_temp_mean_i)
    print(mean_for_row)




    '''
    # Sort the list of tuples by the first element (i) (row)
    sorted_tuples = sorted(bone_coordinates, key=lambda x: x[0])

    # Group tuples based on the same value of i (row)
    grouped_tuples = {}
    for key, group in groupby(sorted_tuples, key=lambda x: x[0]):
        grouped_tuples[key] = list(group) #ogni tuple può avre numeri di elemti variabile in base a quante pixel osso sono state detected in quella riga
    
    list_temp_mean = []
    for k,v in grouped_tuples.items():
        s=0
        for tup in v: #per tuple in v (che rappresenta il vettore di tutte le tuple ordinate per stessa riga)
            s+=tup[1]
        temp_mean = s/len(v)
        list_temp_mean.append(temp_mean)
    mean_for_row=sum(list_temp_mean)/len(list_temp_mean)
    print(mean_for_row)
    '''
            
    
    # (TO-DO) Considerare punti del contorno

    # (TO-DO) Per ogni riga calcolare la media delle colonne
    # es. riga 30: (30,41), (30,42), (30,43), (30,44), (30,45) -> media delle colonne = 43
    # Per ogni colonna calcolare la media delle righe
    # es. colonna 50: (31,50), (32,50), (33,50), (34,50), (35,50) -> media delle righe = 33
                

    # Centroid computation
    centroid_i, centroid_j = 0, 0
    for i,j in bone_coordinates:
        centroid_i += i
        centroid_j += j

    centroid_i /= len(bone_coordinates)
    centroid_j /= len(bone_coordinates)
    centroid_i, centroid_j = int(centroid_i), int(centroid_j)

    print(centroid_i)
    print(centroid_j)
    
    # Bone patch extraction
    assert output_dim[0] == output_dim[1] # Make sure patch is squared
    if output_dim[0] % 2 == 1:
        bone_patch = image[ centroid_i-int(output_dim[0]/2):centroid_i+int(output_dim[0]/2)+1, centroid_j-int(output_dim[1]/2):centroid_j+int(output_dim[1]/2)+1 ]
    else:
        bone_patch = image[ centroid_i-int(output_dim[0]/2):centroid_i+int(output_dim[0]/2), centroid_j-int(output_dim[1]/2):centroid_j+int(output_dim[1]/2) ] 

    return bone_patch


def image_equalization(image: np.ndarray) -> np.ndarray:
    """Image equalization."""

    rescaled_image = (np.maximum(image,0)/image.max())*255 # float pixels
    final_image = np.uint8(rescaled_image) # integers pixels

    return final_image
    


def convert_matrix2png(image, output_path: str, desired_size: Tuple[int] = None):
    """Convert a numpy matrix to png and save to disk."""
    
    if desired_size != None: #TO-DO
        pass

    final_image = Image.fromarray(image)
    final_image.save(output_path)

    return final_image



def main():
    print("Single image preprocessing")

    file_path = os.path.join(os.getcwd(), "data/dicom_data/Pazienti_Settici/Patient1/DX5KWJEJ/KEMJG33A/I1240000")

    dicom_data = pydicom.dcmread(file_path)
    output_file_path1 = os.path.join(os.getcwd(), "data/temp/temp1.png")
    output_file_path2 = os.path.join(os.getcwd(), "data/temp/temp2.png")

    if 'PixelData' in dicom_data and 'RescaleSlope' in dicom_data and 'RescaleIntercept' in dicom_data:
        pixel_array = dicom_data.pixel_array
        height, width = pixel_array.shape
        
        if (height, width) == (512,512): # Filter only axial tomographies

            image_matrix = convert_dcm2matrix(file_path)
            convert_matrix2png(image_equalization(image_matrix), output_file_path1)

            if hounsfield_hp_detection(image_matrix, 
                                       float(dicom_data.RescaleSlope), 
                                       float(dicom_data.RescaleIntercept), 
                                       threshold=3000):
                print("Prothesis detected!")
                image_bone = bone_extraction(image_matrix, 
                                             float(dicom_data.RescaleSlope), 
                                             float(dicom_data.RescaleIntercept), 
                                             threshold=1000, 
                                             output_dim=(188,188))
                
                image_bone_equalized = image_equalization(image_bone)
                
                convert_matrix2png(image_bone_equalized, output_file_path2)

    

if __name__ == "__main__":
    main()

