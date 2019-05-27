# Jonah Siekmann, Jesse Chick, David Skeen
# Assignment 4, Question 2, Part 2
# Principal Component Analysis

# Plot the mean image, and each of the top ten eigen-vectors.
# To make the image for eigen-vectors viewable, you should 
# re-scale each eigenvector by its maximum value. Inspect the 
# resulting images. What do you think they each capture?


import numpy as np
import matplotlib.pyplot as plt
import math

from pca_1 import pca
from kmeans import load_data

def show_vec_as_img(vec, title):
  raw = np.zeros((28, 28, 3))

  vec = np.real(vec) # discard imaginary component if one exists
  vecmin = np.min(vec)
  vecmax = np.max(vec)
  vecrange = (vecmax - vecmin)
  vec = (vec - vecmin) / vecrange

  raw[:,:,0] = np.reshape(vec, (28, 28))
  raw[:,:,1] = raw[:,:,0]
  raw[:,:,2] = raw[:,:,1]

  plt.imshow(raw)
  plt.title(title)
  plt.show()

if __name__ == '__main__':
  data = load_data('./data/p4-data.txt', to_float32=True)

  mean_raw = np.mean(data, axis=0)
  show_vec_as_img(mean_raw, "MEAN IMAGE")

  _, eigenvectors = pca(data)

  for idx, vec in enumerate(eigenvectors, 1):
    show_vec_as_img(vec, "NORMALIZED EIGENVECTOR NUMBER {:2d}".format(idx))
