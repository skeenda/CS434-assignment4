# Jonah Siekmann, Jesse Chick, David Skeen
# Assignment 4, Question 2, Part 1
# Principal Component Analysis

# Problem statement
# Implement Principal Component Analysis for dimension reduction.
# Specifically, your program needs to compute the mean and covariance 
# matrix of the data, and compute the top ten eigen-vectors with ten
# largest eigen-values of the Covariance matrix (you can use existing 
# functions in numpy to compute the eigen-values and eigen-vectors).
# Report the eigenvalues in decreasing order.

import numpy as np
import matplotlib.pyplot as plt

from kmeans import load_data

def pca(data, top_b=10):
  # compute the mean of the data
  mu = np.mean(data)
  centered = data - mu

  # compute the covariance of the data
  sigma = np.cov(centered, rowvar=0)
  
  # get a vector of eigenvalues and matrix of eigenvectors
  lmbda, w = np.linalg.eig(np.mat(sigma))

  # sort in descending order
  idx = lmbda.argsort()[::-1]

  # apply the new indexing and transpose the eigenvector matrix
  lmbda = lmbda[idx]
  w = w[idx].T

  # use top eigenvalues if have less than ten
  if len(lmbda) < 10:
    return lmbda, w

  # if not, return the top ten eigenvalues
  else:
    return lmbda[:10], w[:10]

if __name__ == '__main__':
  data = load_data('./data/p4-data.txt', to_float32=True)
  vals, w = pca(data)
  
  print("Eigenvalues in descending order:")

  for idx, val in enumerate(vals, 1):
    print("{:3d}: {:5.3f}".format(idx, np.real(val)))

