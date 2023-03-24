import numpy as np
import emnist
import cv2 as cv
from Utils import listArrayToString

# Load the EMNIST dataset
emnist_data = emnist.extract_training_samples('letters')

# Separate the images and labels
# print(emnist_data[2])
emnist_images = emnist_data[0]
emnist_labels = emnist_data[1]

# Print the shape of the data
print('EMNIST images shape:', emnist_images.shape)
print('EMNIST labels shape:', emnist_labels.shape)

# Get dimensions of images
width, heigth = emnist_images[0].shape

# For scaling up the image
scale_factor = 20
amountLimiter = 40000

newWidth = int(scale_factor * width)
newHeight = int(scale_factor * heigth)


def writeIntoFile():
    with open('Beginning-Phase/Digits.txt', 'w') as file:
        amount = amountLimiter if amountLimiter < emnist_images.shape[
            0] else emnist_images.shape[0]
        for i in range(amount):
            im = emnist_images[i]
            # label for characters
            imLabel = chr(emnist_labels[i] + ord('a') - 1)
            # label for numbers
            # imLabel = str(emnist_labels[i])
            newMat = im.flatten()
            newMatTxt = listArrayToString(newMat)
            res = imLabel + "-" + newMatTxt + '\n'
            file.write(res)
# writeIntoFile()


for i in range(1000):
    im = emnist_images[i]
    img = cv.resize(im, (newWidth,newHeight))
    # print for letters
    print(chr(emnist_labels[i] + ord('a') - 1))
    # print for digits
    # print(emnist_labels[i])

    # Display the image with text
    cv.imshow('Image and Text', img)

    # Wait for a key press and then close the window
    cv.waitKey(0)
    k = cv.waitKey(0)
    if k == ord('s') or k == ord('S'):  # "S" or "s" key
        break


cv.destroyAllWindows()
