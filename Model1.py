import tensorflow as tf
#import tensorflow_probability as tfp
import numpy as np
from i_vec_layer import i_vec_layer
from keras.layers import Input
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Flatten, concatenate, GlobalMaxPooling1D, Lambda, CuDNNLSTM, TimeDistributed, BatchNormalization, Add, Reshape,UpSampling1D, SeparableConv1D
import h5py
import time
from keras.models import Model
from keras.utils import Sequence
import keras.callbacks
import keras.backend as K
from keras.initializers import Identity, Constant,RandomNormal, Zeros
import os
import scipy.io


def GetModel():
        F=39;
        inp=Input(shape=(None,F));
        #dnn_1=TimeDistributed(Dense(64,activation='relu'))(inp)
        #dnn_2=TimeDistributed(Dense(64,activation='relu'))(dnn_1)
        #dnn_3=TimeDistributed(Dense(64,activation='relu'))(dnn_2)
        dnn_5=TimeDistributed(Dense(39,activation='linear'))(inp)
	
        return(inp,dnn_5)
