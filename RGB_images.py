# importing necessary libraries
import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import eig
from numpy.linalg import eigh
from matplotlib.image import imread
import matplotlib.pyplot as plt
import math


def get_SVD(A):  # function to get SVD values of given matrix
    (n, m) = A.shape  # finding the dimensions of matrix
    (sigma_square, U) = eigh(A @ A.T)  # eigh function to calculate eigen values and vectors for symmetric matrix
    deter = np.linalg.det(U)  # calculating determinant
    deter_sq = deter ** 2
    K = deter_sq ** (1 / (2 * n))  # finding K to divide U to make it orthogonal
    U = U / K
    index = sigma_square.argsort()[::-1]  # finding indeces in sorted order
    sigma_square = sigma_square[index]  # making sigma square sorted
    U = U[:, index]  # arranging eigen vectors according to sorted sigma_square
    sigma = [0] * n  # finding sigma out of sigma square
    for i in range(n):
        sigma[i] = math.sqrt(sigma_square[i])
    S = np.diag(sigma)
    VH = inv(S) @ inv(U) @ A  # finding VH from Sigma and U matrices
    return (U, S, VH)  # returning the values


# reading the image and representing the original image
I = imread(r"C:\Users\HEMA\OneDrive\Documents\PRML\Assnment1\assignment1_rgb.jpg")
n, m, a = I.shape
print(n, m, a)
plt.imshow(I)
plt.title('original image')
plt.show()

I = np.array(I, dtype=np.float64)  # representing data in float64
I = I / 255  # normalizing the image intensity
red = I[:, :, 0]  # separating the RGB components
green = I[:, :, 1]
blue = I[:, :, 2]
plt.imshow(green)
(n, m) = red.shape  # finding the dimensions of original image
RU, RS, RVH = np.linalg.svd(red)  # getting the required matrices from SVD function
RS = np.diag(RS)
GU, GS, GVH = np.linalg.svd(green)  # getting the required matrices from SVD function
GS = np.diag(GS)
BU, BS, BVH = np.linalg.svd(blue)  # getting the required matrices from SVD function
BS = np.diag(BS)

# Reconstructing the images with various rank
# list containing k values that is number of top columns taken
num_of_columns = [1, 2, 4, 8, 13, 25, 30, 35, 41, 49, 60, 66, 225]
i = 0
# reconstructing the image using SVD taking top k columns
for k in num_of_columns:
    approx_RI = RU[:, :k] @ RS[0:k, :k] @ RVH[:k, :]  # calculating approximate R matrix taking top k columns from SVD
    approx_GI = GU[:, :k] @ GS[0:k, :k] @ GVH[:k, :]  # calculating approximate G matrix taking top k columns from SVD
    approx_BI = BU[:, :k] @ BS[0:k, :k] @ BVH[:k, :]  # calculating approximate B matrix taking top k columns from SVD
    approx_I = np.zeros((n, m, a))  # reconstructing the image
    approx_I[:, :, 0] = approx_RI
    approx_I[:, :, 1] = approx_GI
    approx_I[:, :, 2] = approx_BI

    error_I = I - approx_I  # calculating error image
    figure, (ax1, ax2) = plt.subplots(1, 2)  # plotting both subplots
    figure.suptitle('reconstructed image from SVD taking first {} columns of U is'.format(k))
    ax1.imshow(approx_I)
    ax1.set_title('reconstructed image')
    ax2.imshow(error_I)
    ax2.set_title('error image')
    plt.show()

