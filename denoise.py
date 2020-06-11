import keras
import cv2
from keras.models import Model
from keras.layers import Dense, Input, Reshape, Flatten, Lambda, Conv2DTranspose 
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
import os


def denoisee(weight_path,input_image):
    autoencoder = keras.models.load_model('denoise_model.h5')
    img=np.array(keras.preprocessing.image.load_img(input_image) )
    img=img.astype('float32')/255.
    decoded_imgs = autoencoder.predict(img)
    cv2.imwrite(os.path.join('static','uploads','denoised_'+decoded_imgs[0]))

