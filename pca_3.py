# Assignment 4, Question 2, Part 3
# Principal Component Analysis

# Problem statement
# Use the top 10 eigen-vectors to project each image to 10 
# dimensions. Identify for each dimension the image that has 
# the largest value in that dimension and plot it. Compare the 
# image with its corresponding eigenvector image. What do you 
# observe? What do you think the reduced 10-dimensional 
# representation is capturing in this case?

# For each dimension, plot the image that has the largest value 
# in that dimension. Include these images in your report.

import numpy as np
import matplotlib.pyplot as plt

from pca_1 import pca
from pca_2 import show_vec_as_img
from kmeans import load_data

if __name__ == '__main__':
  data = load_data('./data/p4-data.txt', to_float32=True)

  # do the principal component analysis, get top 10 eigenvectors
  eigenvalues, eigenvectors = pca(data)

  # retrieve the principal components from the original data
  principal_components = np.matmul(eigenvectors, data.T).T

  # determine which image had the largest dot-product value for each dimension
  largest_values = []
  for i in range(len(eigenvectors)):
    largest_values.append(np.argmax(principal_components[:,i]))

  best_images = data[largest_values]

  # print the image corresponding to each dimension, and the eigenvector for that dimension
  for idx, (eigenvector, image) in enumerate(zip(eigenvectors, best_images), 1):
    show_vec_as_img(image, "BEST-MATCHING IMAGE ON DIMENSION {:3d}".format(idx))
    show_vec_as_img(eigenvector, "NORMALIZED EIGENVECTOR FOR DIMENSION {:3d}".format(idx))
