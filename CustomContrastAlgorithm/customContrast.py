# customContrast.py --> An attempt to handle shadows better when doing contrast
# November 18 2024 - Iya - Created but creating artifacts
# November 25 2024 - Iya - Redid method (problem was mixing up that 255 is lighter)

import cv2
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bars

def customContrast(image, subsectionSizes):

    height, width = image.shape
    
    newImage = np.copy(image)
    
    
    for n in subsectionSizes:

        # Loop over the image in n by n subsections
        # Use tqdm to show the progress of looping over height
        for i in tqdm(range(0, height, n), desc=f"Rows (n={n})", unit="row"):
            for j in range(0, width, n):
                
                sub_height = min(n, height - i)
                sub_width = min(n, width - j)

                subsection = image[i:i + sub_height, j:j + sub_width]

                tipPoint = np.mean(subsection) * 0.80

                # make changes to pixels based on the tipPoint
                for y in range(sub_height):
                    for x in range(sub_width):
                        pixel_value = subsection[y, x]
                        if pixel_value > tipPoint:
                            subsection[y, x] = min(pixel_value * 1.25, 255) # make it lighter
                        else:
                            subsection[y, x] = max(pixel_value * 0.75, 0) # make it darker

                # Add it back into the new image
                newImage[i:i + sub_height, j:j + sub_width] = subsection

        # Save intermediate image
        cv2.imwrite(f"SubectionOfSize{n}.jpg", newImage)


    return newImage


# Test Code

address = "TheImage"

image = cv2.imread(f'{address}.jpg', cv2.IMREAD_GRAYSCALE) # GreyScale

subsectionSizes = [20, 50, 100, 200, 500]

finalImage = customContrast(image, subsectionSizes)

cv2.imwrite(f'{address}NEW.jpg', finalImage)

