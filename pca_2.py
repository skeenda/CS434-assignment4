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
#from PIL import Image, ImageDraw
from pca_1 import pca

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

def show_vec_as_img(vec, title):
  raw = np.zeros((28, 28, 3))
  raw[:,:,0] = np.reshape(vec, (28, 28))
  raw[:,:,1] = raw[:,:,0]
  raw[:,:,2] = raw[:,:,1]

  plt.imshow(raw)
  plt.title(title)
  plt.show()

if __name__ == '__main__':
  data = load_data('./data/p4-data.txt')
  print(np.max(data))
  mean_raw = np.mean(data, axis=0)
  show_vec_as_img(mean_raw, "mean image")
  _, eigenvectors = pca(data)
  for idx, vec in enumerate(eigenvectors, 1):
    vecmax = np.max(vec)
    show_vec_as_img(vec/vecmax, "eigenvector {:2d}".format(idx))
