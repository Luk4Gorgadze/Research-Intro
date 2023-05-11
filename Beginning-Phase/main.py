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
    # Start the timer
    start_time = time.time()
    while RUN:
        # Calculate the elapsed time
        elapsed_time = time.time() - start_time

        # Print the elapsed time in seconds
        print("Elapsed time: {:.2f} seconds".format(elapsed_time),"--","Iteration:",ITERATION)
        time.sleep(5)

t = threading.Thread(target=write_time)
t.start()

MODE = 'digits'
# Load the EMNIST dataset
emnist_data = emnist.extract_training_samples(MODE)

# Separate the images and labels
emnist_images = emnist_data[0]
emnist_labels = emnist_data[1]

# Print the shape of the data
# print('EMNIST images shape:', emnist_images.shape)
# print('EMNIST labels shape:', emnist_labels.shape)

# Get dimensions of images
width, heigth = emnist_images[0].shape

# For scaling up the image
scale_factor = 20
amountLimiter = 10

newWidth = int(scale_factor * width)
newHeight = int(scale_factor * heigth)


data = TransformImagesToData(MODE,amountLimiter,emnist_images,emnist_labels,True,False)
# print(len(data),len(data[0]))
data = np.array(data)

# Visualise data
def plotData(data):
    # Create 2D grid of X and Y coordinates
    x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

    # Flatten matrix and use as Z coordinate
    z = data.flatten()

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.scatter(x, y, z)
    ax.plot_surface(x, y, data)

    # Label axes
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    ax.set_zlabel('Value')

    # Show plot
    plt.show()
    # END OF PLOTING

plotData(data)

# print(data)
# print(data[0])
# Us = [data[0]]
Us = [[.5 for i in range(len(data[0]))]]

# print(Us)
# U - data
# V - smooth

# Uncomment these 2 lines to see images
# showImages(MODE,amountLimiter,emnist_images,emnist_labels,newWidth,newHeight,data)
# cv.destroyAllWindows()

# Create tridiagonal matrix 
dataLength = len(data[0])
A = np.zeros((dataLength,dataLength))
# print(A)
dt = 0.1
h = 1
s = dt/h
A[0][0] = 10
A[dataLength - 1][dataLength - 1] = 10
y = np.zeros(dataLength)
# print(y)
epsilon = 1e-5
# Iterate through images
for t in range(1,amountLimiter):
    ITERATION += 1 
    for i in range(1,dataLength-1):

        v = smooth(t,i)
        # vm1 = smooth(t-1,i-1)
        # vp1 = smooth(t-1,i+1)
        # V_j_p = 1/2 * (v + abs(v))
        # V_j_m = 1/2 * (v - abs(v))
        # V_jp1 = 1/2 * (vp1 - abs(vp1))
        # V_jm1 = 1/2 * (vm1 + abs(vm1))

        vjp1 = smooth(t+1,i)
        vjm1 = smooth(t-1,i)
        V_j_p = 1/2 * (v + abs(v))
        V_j_m = 1/2 * (v - abs(v))
        V_jp1m = 1/2 * (vjp1 - abs(vjp1))
        V_jm1p = 1/2 * (vjm1 + abs(vjm1))
        # V_j_p = 1/2 * (v + (v*v + epsilon*epsilon)**(1/2))
        # V_j_m = 1/2 * (v - (v*v + epsilon*epsilon)**(1/2))
        # V_jp1 = 1/2 * (vp1 - (vp1*vp1 + epsilon*epsilon)**(1/2))
        # V_jm1 = 1/2 * (vm1 + (vm1*vm1 + epsilon*epsilon)**(1/2))

        
        c = s*V_jm1p
        b = s*V_jp1m
        a = 1 + s*V_j_p - s*V_j_m

        A[i][i-1] = -c
        A[i][i] = a
        A[i][i+1] = b
        
    for j in range(dataLength):
        y[j] = Us[t-1][j]
    # print(y,ITERATION)
    c = solve(A,y)
    # print(ITERATION,c,"\n")
    Us.append(c)

# print(Us)
# for i in range(len(Us)):
    # print(len(Us[i]),len(Us))
    # print(Us[i])
# print(len(Us),len(Us[0]))
# print(Us)
plotData(np.array(Us))
RUN = False


