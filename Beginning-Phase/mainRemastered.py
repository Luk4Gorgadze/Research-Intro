import numpy as np
import emnist
import cv2 as cv
from Utils import TransformImagesToData,showImages
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
# t.start()

MODE = 'digits'

emnist_data = emnist.extract_training_samples(MODE)

# # Separate the images and labels
# emnist_images = emnist_data[0]
# emnist_labels = emnist_data[1]

# # Get dimensions of images
# width, heigth = emnist_images[0].shape

# # For scaling up the image
# scale_factor = 20
# amountLimiter = 10

# newWidth = int(scale_factor * width)
# newHeight = int(scale_factor * heigth)


# data = TransformImagesToData(MODE,amountLimiter,emnist_images,emnist_labels,True,False)
# data = np.array(data)

# # Checkpoint - Now I have all images transformed into 1 dimensional arrays
# # I need to solve continuity equation for each such image


# Us = [data[0]]

# # Create tridiagonal matrix 
# dataLength = len(data[0])
# A = np.zeros((dataLength,dataLength))
# dt = 0.1
# h = 1
# s = dt/h
# A[0][0] = 0
# A[dataLength - 1][dataLength - 1] = 0
# y = np.zeros(dataLength)
# epsilon = 1e-5
# times = [i * dt for i in range(100)]
# for t in times:
#     for i in range(1,dataLength-1):
#         v = smooth(t,i)
#         vjp1 = smooth(t+1,i)
#         vjm1 = smooth(t-1,i)
#         V_j_p = 1/2 * (v + abs(v))
#         V_j_m = 1/2 * (v - abs(v))
#         V_jp1m = 1/2 * (vjp1 - abs(vjp1))
#         V_jm1p = 1/2 * (vjm1 + abs(vjm1))

#         c = s*V_jm1p
#         b = s*V_jp1m
#         a = 1 + s*V_j_p - s*V_j_m

#         A[i][i-1] = -c
#         A[i][i] = a
#         A[i][i+1] = b

#     for j in range(dataLength):
#         y[j] = Us[t-1][j]
#     c = solve(A,y)
#     Us.append(c)


# RUN = False


