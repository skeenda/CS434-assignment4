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

def load_data(path, to_float32=False):
    # Loads data from the path provided as 8-bit pixel values.
    data = np.loadtxt(path, dtype=np.uint8, delimiter=',')
    # If desired, convert to [0, 1] - adds overhead, but makes SSEs easier to
    #   interpret, w/r/t the order of magnitude observed in the SSEs. In our
    #   report, all figures were generated via data of type np.float32, [0, 1].
    if to_float32:
        data = data.astype(np.float32)
        data /= 255
    return data

def pca(data, top_b=10):
  # compute the mean of the data mu = 0.5 * 1/n sum(x)
  # just for fun, np.cov actually does this for us
  mu = np.mean(data, axis=0)
  diff = data - mu

  # compute the covariance of the data
  sigma = np.cov(diff.T)

  # get the a vector of eigenvalues and matrix of eigenvectors
  lmbda, w = np.linalg.eigh(sigma)

  # use top eigenvalues if have less than ten
  if len(lmbda) < 10:
    return lmbda, w

  # if not, return the top ten eigenvalues
  else:
    return lmbda[:10], w[:10]

if __name__ == '__main__':
  data = load_data('./data/p4-data.txt')
  vals, _ = pca(data)
  print("Eigenvalues in descending order:")
  for idx, val in enumerate(vals, 1):
    print("{:3d}: {:16.15f} ({})".format(idx, val, val))

