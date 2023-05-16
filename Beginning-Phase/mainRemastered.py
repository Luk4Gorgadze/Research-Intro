import numpy as np
import emnist
import cv2 as cv
from Utils import TransformImagesToData,showImages,contEquation,plotData
from SmoothFunctions import smooth
from scipy.linalg import solve
import threading
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ITERATION = 0
RUN = True
def write_time():
    start_time = time.time()
    while RUN:
        elapsed_time = time.time() - start_time
        print("Elapsed time: {:.2f} seconds".format(elapsed_time),"--","Iteration:",ITERATION)
        time.sleep(5)

t = threading.Thread(target=write_time)
t.start()

MODE = 'digits'

emnist_data = emnist.extract_training_samples(MODE)

# # Separate the images and labels
emnist_images = emnist_data[0]
emnist_labels = emnist_data[1]

# # Get dimensions of images
width, heigth = emnist_images[0].shape

# # For scaling up the image
scale_factor = 20
amountLimiter = 200

newWidth = int(scale_factor * width)
newHeight = int(scale_factor * heigth)


data = TransformImagesToData(MODE,amountLimiter,emnist_images,emnist_labels,True,False)
data = np.array(data)

# # Checkpoint - Now I have all images transformed into 1 dimensional arrays
# # I need to solve continuity equation for each such image


Us = contEquation(data,150,smooth)
plotData(Us)


RUN = False


