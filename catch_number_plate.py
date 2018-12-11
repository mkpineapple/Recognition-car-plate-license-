# @ Ke Ma for final project
# @contributor: qiqi jiryi

import numpy as np
import cv2
import imutils
import os
import numpy
from skimage import exposure


def process_image(imagePath):

    # Read the image file
    image = cv2.imread(imagePath)
    # Resize the image - change width to 500
    image = imutils.resize(image, width=500)

    # Display the original image
    # cv2.imshow("Original Image", image)

    # RGB to Gray scale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("1 - Grayscale Conversion", gray)

    # Noise removal with iterative bilateral filter(removes noise while preserving edges)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    # cv2.imshow("2 - Bilateral Filter", gray)

    # Find Edges of the grayscale image
    edged = cv2.Canny(gray, 170, 200)
    # cv2.imshow("4 - Canny Edges", edged)

    # Find contours based on Edges
    (new, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] # Sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
    NumberPlateCnt = None # We currently have no Number plate contour

    # loop over our contours to find the best possible approximate contour of number plate
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx # This is our approx Number Plate Contour
            break
            
    # split
    imagePath = imagePath.split("/")
    imagePath = imagePath[1].split(".")
    imagePath = imagePath[0]
    imagePath = "Imaged-detected " + imagePath
    
    try:
        cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)
        cv2.imwrite('Images-detected/%s.png' % imagePath[:imagePath.rfind('.')], image)
        cv2.imwrite(("Images-detected/" + imagePath + ".png"), image)
        idx = 0
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w > 30 and h > 30 and idx == 0:
                idx += 1
                new_img = image[y:y + 1.2*h, x:x + w]
                cv2.imwrite(("Crops/" + imagePath + ".png"), new_img)
    except Exception:
        print("Detection fails")
        idx = 0
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w > 30 and h > 30 and idx == 0:
                idx += 1
                new_img = image[y:y + h, x:x + w]
                cv2.imwrite(("Crops/" + imagePath + ".png"), new_img)
        image = cv2.imread(imagePath)

    return image

if __name__ == '__main__':
    for filename in os.listdir("Images"):
        path = "Image-detected {}".format(filename)
        imagePath = 'Images/'+filename
        image = process_image(imagePath)
        # cv2.imwrite('Images-detected/%s.png' % path[:path.rfind('.')], image)
