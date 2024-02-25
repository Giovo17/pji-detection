from PIL import Image, ImageOps

def equalize_hist_pillow(image):
    # Apply histogram equalization
    equalized_image = ImageOps.equalize(image)
    
    return equalized_image

# Example usage:
input_image = Image.open('/Users/gabrieletuccio/Developer/GitHub/pji-detection/data/temp/temp1.png')
output_image = equalize_hist_pillow(input_image)
output_image.show()
