from scipy.linalg import solve
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def from2Dto1DHorizontal(mat):
    rows, cols = mat.shape
    res = []
    for row in range(rows):
        for col in range(cols):
            res.append(mat[row][col]/255)

    return np.array(res)

def from2Dto1DVertical(mat):
    rows, cols = mat.shape
    res = []
    for col in range(cols):
        for row in range(rows):
            res.append(mat[row][col]/255)

    return np.array(res)


def listArrayToString(narray):
    res = ""
    for i in narray:
        res += str(narray[i]) + "|"
    res = res[:len(res)-1]
    return res


def TransformImagesToData(MODE, amountLimiter, emnist_images, emnist_labels,horizontal,writeToFile):
    """Transforms 2D images into 1D, writes them into text files and returns list of such arrays"""
    path = ""
    resList = []
    if MODE == "letters":
        path = "Beginning-Phase/Letters.txt"
    elif MODE == "digits":
        path = "Beginning-Phase/Digits.txt"
    with open(path, 'w') as file:
        amount = amountLimiter if amountLimiter < emnist_images.shape[
            0] else emnist_images.shape[0]
        for i in range(amount):
            im = emnist_images[i]
            if MODE == "letters":
                # label for characters
                imLabel = chr(emnist_labels[i] + ord('a') - 1)
            elif MODE == "digits":
                # label for numbers
                imLabel = str(emnist_labels[i])

            newMat = None

            if horizontal:
                newMat = from2Dto1DHorizontal(im)
            else:
                newMat = from2Dto1DVertical(im)
            resList.append(newMat)

            if writeToFile:
                newMatTxt = listArrayToString(newMat)
                res = imLabel + "-" + newMatTxt + '\n'
                file.write(res)
    return resList


def showImages(MODE, amountLimiter, emnist_images, emnist_labels, width, height,data):
    """Shows images and prompts out what symbol this picture corresponds to in terminal"""
    for i in range(amountLimiter):
        im = emnist_images[i]
        img = cv.resize(im, (width, height))
        if MODE == "letters":
            # print for letters
            print(chr(emnist_labels[i] + ord('a') - 1))
        elif MODE == "digits":
            # print for digits
            print(emnist_labels[i])

        # Display the image with text
        cv.imshow('Image and Text', img)
        plotGraphs(data[i])

        # Wait for a key press and then close the window
        cv.waitKey(0)
        k = cv.waitKey(0)
        if k == ord('s') or k == ord('S'):  # "S" or "s" key
            break

def plotGraphs(arrData):
    Xs = [x for x in range(len(arrData))]
    Ys = [y for y in arrData]
    # Create a line plot
    plt.plot(Xs, Ys)
    # plt.scatter(Xs, Ys, s=10, marker='o')

    # Add title and axis labels
    plt.title("Line Graph")
    plt.xlabel("X-axis label")
    plt.ylabel("Y-axis label")

    # Show the plot
    plt.show()
   
def contEquation(data,imageIndex,smooth):
    Us = [data[imageIndex]]
    # # Create tridiagonal matrix 
    dataLength = len(data[0])
    A = np.zeros((dataLength,dataLength))
    dt = 0.01
    h = 1
    s = dt/h
    A[0][0] = 1
    A[dataLength - 1][dataLength - 1] = 1
    y = np.zeros(dataLength)
    epsilon = 1e-5
    times = [i * dt for i in range(100)]
    for t in range(1,len(times)-1):
        for i in range(1,dataLength-1):
            tt = times[t]
            tp = times[t+1]
            tm = times[t-1]

            v = smooth(tt,i)

            vjp1 = smooth(tt,i+1)
            vjm1 = smooth(tt,i-1)

            # V_j_p = 1/2 * (v + abs(v))
            # V_j_m = 1/2 * (v - abs(v))
            # V_jp1m = 1/2 * (vjp1 - abs(vjp1))
            # V_jm1p = 1/2 * (vjm1 + abs(vjm1))

            V_j_p = 1/2 * (v + (v**2 + epsilon**2)**(1/2))
            V_j_m = 1/2 * (v - (v**2 - epsilon**2)**(1/2))
            V_jp1m = 1/2 * (vjp1 - (vjp1**2 - epsilon**2)**(1/2))
            V_jm1p = 1/2 * (vjm1 + (vjm1**2 + epsilon**2)**(1/2))


            c = s*V_jm1p
            b = s*V_jp1m
            a = 1 + s*V_j_p - s*V_j_m
            # print(b)
            A[i][i-1] = -c
            A[i][i] = a
            A[i][i+1] = b
        y[0] = 0
        y[dataLength-1] = 0
        for j in range(1,dataLength-1):
            y[j] = Us[t-1][j]
        # print(A[0])
        c = solve(A,y)
        Us.append(c)
    return np.array(Us)

def plotData(data):
    x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    z = data.flatten()
    
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