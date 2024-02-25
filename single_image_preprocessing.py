import os
import numpy as np
import pydicom
from PIL import Image, ImageOps, UnidentifiedImageError
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
            if image[i,j] * slope + intercept > threshold:
                return True

    return False


def bone_extraction(image: np.ndarray, slope: float, intercept: float, metal_threshold: int, bone_threshold: int, output_dim: Tuple[int]) -> np.ndarray:
    """Bone patch extraction."""

    # Image halving retaining only the part which contains the prothesis
    image_left = image[0:image.shape[0], 0:int(image.shape[1]/2)]
    image_right = image[0:image.shape[0], int(image.shape[1]/2)+1:image.shape[1]]
    metal_half_detector = False # True = metal in the left part, False = metal in the right part
    
    for i in range(len(image_left)):
        if metal_half_detector:
            break
        for j in range(len(image_left[i])):
            if image[i,j] * slope + intercept > metal_threshold:
                metal_half_detector = True
                break
    
    image = image_left if metal_half_detector else image_right


    # Bone coordinates detection
    bone_coordinates = []
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i,j] * slope + intercept > bone_threshold:
                bone_coordinates.append((i,j))


    # Centroid computation
    centroid_i, centroid_j = 0, 0
    for i,j in bone_coordinates:
        centroid_i += i
        centroid_j += j

    centroid_i /= len(bone_coordinates)
    centroid_j /= len(bone_coordinates)
    centroid_i, centroid_j = int(centroid_i), int(centroid_j)

    print(f"Prothesis centroid x: {centroid_i}")
    print(f"Prothesis centroid y: {centroid_j}")
    

    # Bone patch extraction
    assert output_dim[0] == output_dim[1] # Make sure patch is squared
    assert output_dim[0] < image.shape[0] # Make sure patch is smaller than image
    assert output_dim[0] < image.shape[1] # Make sure patch is smaller than image

    addon = 1 if output_dim[0] % 2 == 1 else 0

    # Exception handling: patch is outside the image
    if centroid_i-int(output_dim[0]/2) < 0:
        if centroid_j-int(output_dim[1]/2) < 0: # Upper left corner
            bone_patch = image[ 0:output_dim[0], 0:output_dim[1] ]
        elif centroid_j+int(output_dim[1]/2)+1 > image.shape[1]: # Upper right coner
            bone_patch = image[ 0:output_dim[0], image.shape[0]-output_dim[0]:image.shape[0] ]
        else: # Upper side without corners
            bone_patch = image[ 0:output_dim[0], centroid_j-int(output_dim[1]/2):centroid_j+int(output_dim[1]/2)+addon ]
    elif centroid_i+int(output_dim[0]/2)+addon > image.shape[0]:
        if centroid_j-int(output_dim[1]/2) < 0: # Lower left corner
            bone_patch = image[ image.shape[0]-output_dim[0]:image.shape[0], 0:output_dim[1] ]
        elif centroid_j+int(output_dim[1]/2)+1 > image.shape[1]: # Lower right corner
            bone_patch = image[ image.shape[0]-output_dim[0]:image.shape[0], image.shape[1]-output_dim[1]:image.shape[1] ]
        else: # Lower side without corners
            bone_patch = image[ image.shape[0]-output_dim[0]:image.shape[0], centroid_j-int(output_dim[1]/2):centroid_j+int(output_dim[1]/2)+addon ]
    else:
        if centroid_j-int(output_dim[1]/2) < 0: # Left side without corners

            bone_patch = image[ centroid_i-int(output_dim[0]/2):centroid_i+int(output_dim[0]/2)+addon, 0:output_dim[1] ]
        elif centroid_j+int(output_dim[1]/2)+1 > image.shape[1]: # Right side without corners
            bone_patch = image[ centroid_i-int(output_dim[0]/2):centroid_i+int(output_dim[0]/2)+addon, image.shape[1]-output_dim[1]:image.shape[1] ]
        else: # Patch all inside the image
            bone_patch = image[ centroid_i-int(output_dim[0]/2):centroid_i+int(output_dim[0]/2)+addon, centroid_j-int(output_dim[1]/2):centroid_j+int(output_dim[1]/2)+addon ]

    
    return bone_patch


def image_equalization(image: np.ndarray) -> np.ndarray:
    """Image equalization."""

    rescaled_pixel_value_image = (np.maximum(image,0)/image.max())*255 # Rescale to 8 bit values
    int_pixel_image = np.uint8(rescaled_pixel_value_image) # Convert pixel value to integer

    if int_pixel_image.mode != 'L': # Convert the image to grayscale if it's not already
        int_pixel_image = int_pixel_image.convert('L')
    equalized_image = ImageOps.equalize(int_pixel_image)  # Apply histogram equalization

    return equalized_image 
    


def convert_matrix2png(image, output_path: str, desired_size: Tuple[int] = None):
    """Convert a numpy matrix to png and save to disk."""
    
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
                                             metal_threshold=3000,
                                             bone_threshold=1000, 
                                             output_dim=(188,188))
                
                image_bone_equalized = image_equalization(image_bone)
                
                convert_matrix2png(image_bone_equalized, output_file_path2)

    

if __name__ == "__main__":
    main()

