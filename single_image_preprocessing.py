import os
import numpy as np
import pydicom
from PIL import Image
from PIL import UnidentifiedImageError
from typing import Tuple


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

    rescaled_image = (np.maximum(im,0)/im.max())*255 # float pixels
    final_image = np.uint8(rescaled_image) # integers pixels

    return final_image



def hounsfield_hp_detection(image: np.ndarray, slope: float, intercept: float, threshold: int) -> bool:
    """Hounsfield based prothesis detection."""

    for i in range(len(image)):
        for j in range(len(image[i])):
            print(image[i,j])
            if image[i,j] * slope + abs(intercept) > threshold:
                return True

    return False


def bone_extraction(image: np.ndarray, slope: float, intercept: float, threshold: int, output_dim: Tuple[int]) -> np.ndarray:
    """Bone patch extraction."""

    # Bone coordinates detection
    bone_coordinates = []
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i,j] * slope + intercept > threshold:
                bone_coordinates.append([(i,j)])

    # Centroid computation
    centroid_i, centroid_j = 0, 0
    for i,j in bone_coordinates:
        centroid_i += i
        centroid_j += j

    centroid_i /= len(bone_coordinates)
    centroid_j /= len(bone_coordinates)
    
    # Bone patch extraction
    assert output_dim[0] == output_dim[1] # Make sure patch is squared
    if output_dim[0] % 2 == 1:
        bone_patch = image[int(centroid_i)-output_dim[0]/2:centroid_i+int(output_dim[0])/2, centroid_j-int(output_dim[0])/2:centroid_j+int(output_dim[0])/2]
    else:
        bone_patch = image[centroid_i-output_dim[0]/2-1:centroid_i+output_dim[0]/2, centroid_j-output_dim[1]/2-1:centroid_j+output_dim[1]/2] 

    return bone_patch


def image_equalization():
    pass



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
    output_file_path = os.path.join(os.getcwd(), "data/temp/temp.png")

    if 'PixelData' in dicom_data and 'RescaleSlope' in dicom_data and 'RescaleIntercept' in dicom_data:
        pixel_array = dicom_data.pixel_array
        height, width = pixel_array.shape
        
        if (height, width) == (512,512): # Filter only axial tomographies

            image_matrix = convert_dcm2matrix(file_path)

            if hounsfield_hp_detection(image_matrix, int(dicom_data.RescaleSlope), int(dicom_data.RescaleIntercept), threshold=3000):
                print("Prothesis detected!")
                image_bone = bone_extraction(image_matrix, int(dicom_data.RescaleSlope), int(dicom_data.RescaleIntercept), threshold=1000)
                
                #image_equalized = image_equalization(image_bone)
                
                #convert_matrix2png(image_equalized)
                convert_matrix2png(image_bone, output_file_path)

    

if __name__ == "__main__":
    main()

