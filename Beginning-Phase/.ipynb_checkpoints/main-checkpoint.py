import numpy as np
import emnist
import cv2 as cv
from Utils import TransformImagesToData,showImages

MODE = 'digits'
# Load the EMNIST dataset
emnist_data = emnist.extract_training_samples(MODE)

# Separate the images and labels
emnist_images = emnist_data[0]
emnist_labels = emnist_data[1]

# Print the shape of the data
print('EMNIST images shape:', emnist_images.shape)
print('EMNIST labels shape:', emnist_labels.shape)

# Get dimensions of images
width, heigth = emnist_images[0].shape

# For scaling up the image
scale_factor = 20
amountLimiter = 100

newWidth = int(scale_factor * width)
newHeight = int(scale_factor * heigth)


data = TransformImagesToData(MODE,amountLimiter,emnist_images,emnist_labels,False)
showImages(MODE,amountLimiter,emnist_images,emnist_labels,newWidth,newHeight,data)






cv.destroyAllWindows()
