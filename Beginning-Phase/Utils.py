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
   