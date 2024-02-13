import os
from PIL import Image
from typing import Tuple

data_path = os.getcwd() + "/data/png_export/"
export_path = data_path + "png_resize_"
export_resized_path = data_path + "/png_resized"

def resize_images(path: str, dimension: Tuple[int]):
    os.makedirs(export_path + str(dimension[0])+'x'+str(dimension[0]), exist_ok=True)
    for root, dirs, files in os.walk(path, topdown=False):
        for image in files:
            im = Image.open(image) 
            new_image = image.resize(dimension)
            new_image.save(image+'.png')


def main(action = 'cropped', desired_size =(333,333)):
    #resize_images(data_path)
    os.makedirs(export_resized_path + '/' + 'cropped' +'_'+ str(desired_size[0])+'x'+str(desired_size[0])+ "/" + 'Patiente10' + "/" + 'abcd' , exist_ok=True)
    

if __name__ == "__main__":
    main()
