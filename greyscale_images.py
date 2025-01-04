# importing necessary libraries
import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import eig
from numpy.linalg import eigh
from matplotlib.image import imread
import matplotlib.pyplot as plt
import math


def get_EVD(A):  # function to get EVD values of given matrix
    (Lambda, X) = eig(A)  # eig function to calculate eigen values and vectors
    index = abs(Lambda).argsort()[::-1]  # finding indeces in sorted order
    Lambda = Lambda[index]  # making lambda sorted
    X = X[:, index]  # arranging eigen vectors according to sorted lambda
    L = np.diag(Lambda)
    X_inverse = inv(L) @ inv(X) @ A  # finding X inverse from lambda and X matrices
    return (X, L, X_inverse, Lambda)  # returning the values


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
I = imread(r"C:\Users\HEMA\OneDrive\Documents\PRML\Assnment1\assignment1_image30.jpg")
plt.imshow(I, cmap='gray')
plt.title('original image')
plt.show()

I = np.array(I, dtype=np.float64)  # representing data in float64
(n, m) = I.shape  # finding the dimensions of original image
U, S, VH = get_SVD(I)  # getting the required matrices from SVD function
X, L, XI, Lambda = get_EVD(I)  # getting the required matrices from EVD function

# Reconstructing the images with various rank
# list containing k values that is number of top columns taken
num_of_columns = [1, 2, 4, 8, 13, 25, 30, 35, 41, 49, 60, 66, 256]
i = 0
# reconstructing the image using SVD and EVD taking top k columns
for k in num_of_columns:
    approx_I = U[:, :k] @ S[0:k, :k] @ VH[:k, :]  # calculating approximate image taking top k columns from SVD
    error_I = I - approx_I  # calculating error image
    figure, (ax1, ax2) = plt.subplots(1, 2)  # plotting both subplots
    figure.suptitle('reconstructed image from SVD taking first {} columns of U is'.format(k))
    ax1.imshow(approx_I, cmap='gray')
    ax1.set_title('reconstructed image')
    ax2.imshow(error_I, cmap='gray')
    ax2.set_title('error image')
    plt.show()

    approx_I = X[:, :k] @ L[0:k, :k] @ XI[:k, :]  # calculating approximate image taking top k columns from EVD
    error_I = I - approx_I  # calculating error image
    figure, (ax1, ax2) = plt.subplots(1, 2)  # plotting both subplots
    figure.suptitle('reconstructed image from EVD taking first {} columns of X is'.format(k))
    ax1.imshow((approx_I).real, cmap='gray')
    ax1.set_title('reconstructed image')
    ax2.imshow((error_I).real, cmap='gray')
    ax2.set_title('error image')
    plt.show()
    i += 1

# initialising the variables with actual image norm value to calculate the norm and x axis values for plotting
x_EVD = [0]  # k values taken for EVD values considering conjugate pairs
x_allk = [0]  # to contain all k values
Fnorm_SVD = [np.linalg.norm(I)]  # norm calculated from SVD
Fnorm_EVD = [np.linalg.norm(I)]  # norm calculated from EVD
Fnorm_allk = [np.linalg.norm(I)]  # norm calculated using EVD but not considering conjugate pairs

for k in range(1, n + 1):
    approxI_SVD = U[:, :k] @ S[0:k, :k] @ VH[:k, :]  # calculating each approximated and error images from SVD
    errorI_SVD = I - approxI_SVD

    approxI_EVD = X[:, :k] @ L[0:k, :k] @ XI[:k, :]  # calculating each approximated and error images from EVD
    errorI_EVD = I - approxI_EVD

    x_allk.append(k)  # adding the k value
    Fnorm_allk.append(np.linalg.norm(errorI_EVD))  # appending the individual norms
    Fnorm_SVD.append(np.linalg.norm(errorI_SVD))

    if Lambda[k - 1].imag <= 0:  # deciding whether to take that particular k or not
        x_EVD.append(k)  # considering conjugate pairs
        Fnorm_EVD.append(np.linalg.norm(errorI_EVD))  # appending the norm

# plotting of norm considering all values of k with increment 1 from SVD values
plt.plot(x_allk, Fnorm_SVD)
plt.title('plotting of norm considering all values of k from SVD values')
plt.xlabel(r'k $\rightarrow$', size=16)
plt.ylabel(r'Norm $\rightarrow$', size=16)
plt.grid()
plt.show()

# plotting of norm considering all values of k except 0 with increment 1 from SVD values
plt.plot(x_allk[1:], Fnorm_SVD[1:])
plt.title('plotting of norm considering all values of k except 0 from SVD values')
plt.xlabel(r'k $\rightarrow$', size=16)
plt.ylabel(r'Norm $\rightarrow$', size=16)
plt.grid()
plt.show()

# plotting of norm considering all values of k with increment 1 from EVD values
plt.plot(x_allk, Fnorm_allk)
plt.title('plotting of norm considering all values of k from EVD values')
plt.xlabel(r'k $\rightarrow$', size=16)
plt.ylabel(r'Norm $\rightarrow$', size=16)
plt.grid()
plt.show()

# plotting of norm considering conjugate pairs from EVD values
plt.plot(x_EVD, Fnorm_EVD)
plt.title('plotting of norm considering conjugate pairs from EVD values')
plt.xlabel(r'k $\rightarrow$', size=16)
plt.ylabel(r'Norm $\rightarrow$', size=16)
plt.grid()
plt.show()

# plotting of norm considering conjugate pairs from EVD values excluding k=0
plt.plot(x_EVD[1:], Fnorm_EVD[1:])
plt.title('plotting of norm considering conjugate pairs from EVD values excluding k=0')
plt.xlabel(r'k $\rightarrow$', size=16)
plt.ylabel(r'Norm $\rightarrow$', size=16)
plt.grid()
plt.show()

# plotting the norms calculated from SVD and EVD and comparing
Fnorm_svd = []  # this is to adjust k values of svd accordingly to evd
for i in x_EVD[1:]:
    Fnorm_svd.append(Fnorm_SVD[i])
plt.plot(x_EVD[1:], Fnorm_EVD[1:], label='EVD')
plt.plot(x_EVD[1:], Fnorm_svd, label='SVD')
plt.title('plotting of norm considering conjugate pairs from EVD values')
plt.xlabel(r'k $\rightarrow$', size=16)
plt.ylabel(r'Norm $\rightarrow$', size=16)
plt.legend()
plt.grid()
plt.show()
