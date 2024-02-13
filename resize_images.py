import os
from PIL import Image
from typing import Tuple

data_path = os.getcwd() + "/data/png_export/"
export_path = data_path + "png_resize_"

def resize_images(path: str, dimension: Tuple[int]):
    os.makedirs(export_path + str(dimension[0])+'x'+str(dimension[0]), exist_ok=True)
    for root, dirs, files in os.walk(path, topdown=False):
        for image in files:
            im = Image.open(image) 
            new_image = image.resize(dimension)
            new_image.save(image+'.png')


def main():
    resize_images(data_path)


if __name__ == "__main__":
    main()
