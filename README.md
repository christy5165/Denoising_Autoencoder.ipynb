# Denoising_Autoencoder.ipynb
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras import layers,models,datasets

#Load the MNIST digits (clean images)
(x_train,_),(x_test,_)=datasets.mnist.load_data()

#Normalize to 0-1 range 
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

# Reshape for the AI (Make sure trian uses train, and test uses test)
x_train = np.reshape(x_train, (len(x_train),28,28,1))
x_test = np.reshape(x_test, (len(x_test), 28,28,1))

print("Data loaded and cleaned!")
