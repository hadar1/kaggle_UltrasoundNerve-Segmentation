import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
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





def get_data():
    K.set_image_data_format('channels_last')
    print(os.listdir("input"))
    path = "input/train/"
    file_list = os.listdir(path)
    file_list[:20]
    reg = re.compile("[0-9]+")
    temp1 = list(map(lambda x: reg.match(x).group(), file_list))
    temp1 = list(map(int, temp1))
    temp2 = list(map(lambda x: reg.match(x.split("_")[1]).group(), file_list))
    temp2 = list(map(int, temp2))
    file_list = [x for _, _, x in sorted(zip(temp1, temp2, file_list))]
    # file_list=file_list[:400]
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

    X = []
    y = []
    for image, mask in zip(train_image, train_mask):
        X.append(np.array(Image.open(path + image)))
        y.append(np.array(Image.open(path + mask)))
    X = np.array(X)
    y = np.array(y)
    print("X shape: {}, y shape: {}".format(X.shape,y.shape))
    mask_df = pd.read_csv("input/train_masks.csv")

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
        # image = Image.open("input/train/" + image_path)
        # mask = Image.open("input/train/" + mask_path)
        #
        # image=image.resize((IMG_HEIGHT, IMG_WIDTH), Image.BICUBIC)
        # mask=mask.resize((IMG_HEIGHT, IMG_WIDTH), Image.BICUBIC)
        image = cv2.imread("input/train/" + image_path,0)
        mask = cv2.imread("input/train/" + mask_path,0)

        image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
        mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)

        X[i] = np.array(image)
        y[i] = np.array(mask)

    X = X[:, :, :, np.newaxis] / 255
    y = y[:, :, :, np.newaxis] / 255

    print("X shape : ", X.shape)
    print("y shape : ", y.shape)
    return X,y,IMG_HEIGHT, IMG_WIDTH

def dice_coef(y_true, y_pred):
    smooth=1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
	
def conv_block(inputs,filters):
    conv = Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    conv = Conv2D(filters, (3, 3), activation='relu', padding='same')(conv)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    return conv, pool


def conv_transpose_blocks(conv, conv2, filter):
    up = Conv2DTranspose(filter, (2, 2), strides=(2, 2), padding='same')(conv2)
    up = concatenate([up, conv], axis=3)
    up = Conv2D(filter, (3, 3), activation='relu', padding='same')(up)
    up = Conv2D(filter, (3, 3), activation='relu', padding='same')(up)
    return up



def get_model(IMG_HEIGHT, IMG_WIDTH):
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

def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])


def main():
    X,y,IMG_HEIGHT, IMG_WIDTH = get_data()

    model = get_model(IMG_HEIGHT, IMG_WIDTH)

    results = model.fit(X, y, validation_split=0.1, batch_size=4, epochs=20)

    sub = pd.read_csv("input/sample_submission.csv")
    test_list = os.listdir("input/test")

    print("The number of test data : ", len(test_list))

    # Sort the test set in ascending order.
    reg = re.compile("[0-9]+")

    temp1 = list(map(lambda x: reg.match(x).group(), test_list))
    temp1 = list(map(int, temp1))

    test_list = [x for _, x in sorted(zip(temp1, test_list))]

    test_list=test_list[:15]

    X_test = np.empty((len(test_list), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
    for i, image_path in enumerate(test_list[:10]):
        image = cv2.imread("input/test/" + image_path,0)
        image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_AREA)
        X_test[i] = np.array(image)

    X_test = X_test[:, :, :, np.newaxis] / 255

    y_pred = model.predict(X_test)

    mask_df = pd.read_csv("input/train_masks.csv")

    width = 580
    height = 420

    temp = mask_df["pixels"][0]
    temp = temp.split(" ")

    mask1 = np.zeros(height * width)
    for i, num in enumerate(temp):
        if i % 2 == 0:
            run = int(num) - 1  # very first pixel is 1, not 0
            length = int(temp[i + 1])
            mask1[run:run + length] = 255

    # Since pixels are numbered from top to bottom, then left to right, we are careful to change the shape
    mask1 = mask1.reshape((width, height))
    mask1 = mask1.T

    subject_df = mask_df[['subject', 'img']].groupby(by='subject').agg('count').reset_index()
    subject_df.columns = ['subject', 'N_of_img']
    subject_df.sample(10)

    pd.value_counts(subject_df['N_of_img']).reset_index()

    rles = []
    for i in range(X_test.shape[0]):
        img = y_pred[i, :, :, 0]
        img = img > 0.5
        img = resize(img, (420, 580), preserve_range=True)
        rle = run_length_enc(img)
        rles.append(rle)
        if i % 100 == 0:
            print('{}/{}'.format(i, X_test.shape[0]), end="\r")


    sub['pixels'] = rles
    sub.to_csv("submission.csv", index=False)

main()
