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
from pca_1 import load_data

def show_vec_as_img(vec, title):
  raw = np.zeros((28, 28, 3))
  raw[:,:,0] = np.reshape(vec, (28, 28)).T
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
    vecmax = np.max(vec)
    #print(vec)
    #print(vec/vecmax)
    show_vec_as_img(vec/vecmax, "EIGENVECTOR NUMBER {:2d}".format(idx))
