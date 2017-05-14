#!/usr/bin/env python3
import os
import csv
import cv2
import numpy as np
import sklearn
import copy
import random
import keras
import tensorflow as tf
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.core import Lambda
from sklearn.model_selection import train_test_split
from skimage import draw
from functools import lru_cache
import matplotlib.pyplot as plt
from helper import *



# Take an array of boards, and array of who won - 0 if computer, 1 if human

model = None

def makeModel():
    global model
    if model != None:
        return
    inputs = keras.layers.Input(shape=(2,3,3))

    output = Flatten()(inputs)
    output = Dense(100, activation='relu')(inputs)
    output = Dropout(0.5)(output)
    output = Dense(50, activation='relu')(output)
    output = Dropout(0.5)(output)
    output = Dense(20, activation='relu')(output)
    output = Dropout(0.5)(output)
    output = Dense(1, activation='relu', use_bias=False)(output)

    model = Model(inputs=inputs, outputs=output)

    tbCallBack = keras.callbacks.TensorBoard(
        log_dir='./log', histogram_freq=1, write_graph=True, write_images=True,
        embeddings_freq=1, embeddings_layer_names=None, embeddings_metadata=None)
    checkpointCallback = keras.callbacks.ModelCheckpoint(
        'model_running.h5', monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='auto', period=1)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2,
        patience=5, min_lr=0.0001)

    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))
    from keras.models import load_model
    #model = load_weights('model_running.h5')

boardgames = []
whowon = []


def train(boardgames, whowon):
    global model
    makeModel()
    model.fit(boardgames, whowon, epochs=100, validation_split=0.2, shuffle=True,
              verbose=1, callbacks=[tbCallBack, checkpointCallback, reduce_lr])

# board[0,:,:] is for computer player.  0 if there's no piece and 1 if there is
# board[1,:,:] is for other player.     0 if there's no piece and 1 if there is
def find_next_best_move(board):
    global model





