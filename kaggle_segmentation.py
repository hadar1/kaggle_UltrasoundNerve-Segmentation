import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import cv2
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.models import Model, Input, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from PIL import Image
import keras
import tensorflow as tf
from keras import backend as K


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_data():
    K.set_image_data_format('channels_last')
    print(os.listdir("../input"))
    path = "../input/train/"
    file_list = os.listdir(path)
    file_list[:20]
    reg = re.compile("[0-9]+")
    temp1 = list(map(lambda x: reg.match(x).group(), file_list))
    temp1 = list(map(int, temp1))
    temp2 = list(map(lambda x: reg.match(x.split("_")[1]).group(), file_list))
    temp2 = list(map(int, temp2))
    file_list = [x for _, _, x in sorted(zip(temp1, temp2, file_list))]
    file_list[:20]
    train_image = []
    train_mask = []
    for idx, item in enumerate(file_list):
        if idx % 2 == 0:
            train_image.append(item)
        else:
            train_mask.append(item)
    print(train_image[:10], "\n", train_mask[:10])
    image1 = np.array(Image.open(path + "1_1.tif"))
    image1_mask = np.array(Image.open(path + "1_1_mask.tif"))
    image1_mask = np.ma.masked_where(image1_mask == 0, image1_mask)
    fig, ax = plt.subplots(1, 3, figsize=(16, 12))
    ax[0].imshow(image1, cmap='gray')
    ax[1].imshow(image1_mask, cmap='gray')
    ax[2].imshow(image1, cmap='gray', interpolation='none')
    ax[2].imshow(image1_mask, cmap='jet', interpolation='none', alpha=0.7)
    X = []
    y = []
    for image, mask in zip(train_image, train_mask):
        X.append(np.array(Image.open(path + image)))
        y.append(np.array(Image.open(path + mask)))
    X = np.array(X)
    y = np.array(y)
    mask_df = pd.read_csv("../input/train_masks.csv")
    mask_df.head()
    # Randomly choose the indices of data used to train our model.
    indices = np.random.choice(range(len(train_image)), replace=False, size=100)
    train_image_sample = np.array(train_image)[indices]
    train_mask_sample = np.array(train_mask)[indices]
    IMG_HEIGHT = 96
    IMG_WIDTH = 96
    X = np.empty(shape=(len(indices), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
    y = np.empty(shape=(len(indices), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
    for i, (image_path, mask_path) in enumerate(zip(train_image_sample, train_mask_sample)):
        # image = plt.imread("../input/train/" + image_path)
        # mask = plt.imread("../input/train/" + mask_path)
        image = Image.open("input/train/" + image_path)
        mask = Image.open("input/train/" + mask_path)
        image.thumbnail((IMG_HEIGHT, IMG_WIDTH), Image.ANTIALIAS)
        mask.thumbnail((IMG_HEIGHT, IMG_WIDTH), Image.ANTIALIAS)

        X[i] = image
        y[i] = mask
    X = X[:, :, :, np.newaxis] / 255
    y = y[:, :, :, np.newaxis] / 255
    print("X shape : ", X.shape)
    print("y shape : ", y.shape)
    return X,y,IMG_HEIGHT, IMG_WIDTH

def conv_block(inputs,filters):
    conv = Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    conv = Conv2D(filters, (3, 3), activation='relu', padding='same')(conv)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    return conv, pool


def conv_transpose_blocks(conv, conv2, filter):
    up = Conv2DTranspose(filter, (2, 2), strides=(2, 2), padding='same')(conv2)
    up = concatenate([up, conv], axis=3)
    conv = Conv2D(filter, (3, 3), activation='relu', padding='same')(up)
    conv = Conv2D(filter, (3, 3), activation='relu', padding='same')(conv)
    return conv



def main():
    X,y,IMG_HEIGHT, IMG_WIDTH = get_data()

    model = get_model(IMG_HEIGHT, IMG_WIDTH)

    results = model.fit(X, y, validation_split=0.1, batch_size=4, epochs=20)

def get_model(IMG_HEIGHT, IMG_WIDTH):
    smooth = 1.
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, 1))
    conv1, pool1 = conv_block(inputs, 32)
    conv2, pool2 = conv_block(pool1, 64)
    conv3, pool3 = conv_block(pool2, 128)
    conv4, pool4 = conv_block(pool3, 256)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv6 = conv_transpose_blocks(conv4, conv5, 256)
    conv7 = conv_transpose_blocks(conv3, conv6, 128)
    conv8 = conv_transpose_blocks(conv2, conv7, 64)
    conv9 = conv_transpose_blocks(conv1, conv8, 32)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    return model


main()