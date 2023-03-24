import numpy as np

def transform2Dto1D(mat):
    rows, cols = mat.shape
    res = []
    for row in range(rows):
        for col in range(cols):
            res.append(mat[row][col])

    return np.array(res)



def listArrayToString(narray):
    res = ""
    for i in narray:
        res += str(narray[i]) + "|"
    res = res[:len(res)-1]
    return res