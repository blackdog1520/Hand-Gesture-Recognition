#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow import keras

get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(1)


# In[2]:





import cnn_utils


# In[ ]:


train_dataset = h5py.File('/train_signs.h5', "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:]) #train set 
train_set_y_orig = np.array(train_dataset["train_set_y"][:]) #train set labels

test_dataset = h5py.File('/test_signs.h5', "r")
test_set_x_orig = np.array(test_dataset["test_set_x"][:]) #test set
test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # test set labels


# In[ ]:


train_set_x_orig.shape


# In[ ]:


train_set_y_orig.shape


# In[ ]:


test_set_x_orig.shape


# In[ ]:


test_set_y_orig.shape


# In[ ]:


color1 = train_set_x_orig[0]
plt.imshow(color1, cmap=plt.get_cmap('gray'))
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_set_x_orig[i], cmap="gray")
    plt.xlabel(train_set_y_orig[i])
plt.show()


# In[ ]:


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    # ITU-R 601-2 LUMA TRANSFORM


# In[ ]:


train_images = []
for i in train_set_x_orig :
  train_images.append(rgb2gray(i))

test_images = []
for i in test_set_x_orig :
  test_images.append(rgb2gray(i))

train_images = np.array(train_images)
test_images = np.array(test_images)


# In[ ]:


print(train_images.shape)
print(test_images.shape)


# In[ ]:


plt.imshow(train_images[0], cmap=plt.get_cmap('gray'))
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap="gray")
    plt.xlabel(train_set_y_orig[i])
plt.show()


# In[ ]:


train_images = train_images / 255.0

test_images = test_images / 255.0


# In[ ]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(64, 64)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


# In[ ]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.fit(train_images, train_set_y_orig, epochs=100)


# In[ ]:


test_loss, test_acc = model.evaluate(test_images,  test_set_y_orig, verbose=2)

print('\nTest accuracy:', test_acc)


# In[ ]:


pred = model.predict(test_images)


# In[ ]:


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap="gray")

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("Pred: {} Conf: {:2.0f}% (True: {})".format(predicted_label,
                                100*np.max(predictions_array),
                                test_set_y_orig[i]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(6))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# In[ ]:


i = 3
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, pred[i], test_set_y_orig, test_images)
plt.subplot(1,2,2)
plot_value_array(i, pred[i],  test_set_y_orig)
plt.show()


# In[ ]:




