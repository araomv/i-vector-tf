from __future__ import print_function
import os
import sys
import random
random.seed(9001)
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Flatten, concatenate, GlobalMaxPooling1D, Lambda, CuDNNLSTM, TimeDistributed, BatchNormalization, Add, Reshape,UpSampling1D, SeparableConv1D
from keras.models import Model
from keras.callbacks import RemoteMonitor
from keras.layers import Bidirectional
from keras.initializers import Constant
import keras.backend as K
import numpy.random
import numpy as np
import keras.callbacks
import scipy.io
import os
import time
from shutil import copyfile
import h5py
from keras.callbacks import TensorBoard
import tensorflow as tf
from keras.models import load_model
from random import randint
import math
import progressbar
import glob
from scipy.signal import hilbert
from keras.utils import Sequence
from scipy.signal import decimate
from keras.constraints import Constraint,UnitNorm
from keras.engine.topology import Layer
import math as m
#import tensorflow_probability as tfp
#tfd = tfp.distributions

def block_diagonal(matrices, dtype=tf.float32):
  r"""Constructs block-diagonal matrices from a list of batched 2D tensors.

  Args:
    matrices: A list of Tensors with shape [..., N_i, M_i] (i.e. a list of
      matrices with the same batch dimension).
    dtype: Data type to use. The Tensors in `matrices` must match this dtype.
  Returns:
    A matrix with the input matrices stacked along its main diagonal, having
    shape [..., \sum_i N_i, \sum_i M_i].

  """
  matrices = [tf.convert_to_tensor(matrix, dtype=dtype) for matrix in matrices]
  blocked_rows = tf.Dimension(0)
  blocked_cols = tf.Dimension(0)
  batch_shape = tf.TensorShape(None)
  for matrix in matrices:
    full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
    batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
    blocked_rows += full_matrix_shape[-2]
    blocked_cols += full_matrix_shape[-1]
  ret_columns_list = []
  for matrix in matrices:
    matrix_shape = tf.shape(matrix)
    ret_columns_list.append(matrix_shape[-1])
  ret_columns = tf.add_n(ret_columns_list)
  row_blocks = []
  current_column = 0
  for matrix in matrices:
    matrix_shape = tf.shape(matrix)
    row_before_length = current_column
    current_column += matrix_shape[-1]
    row_after_length = ret_columns - current_column
    row_blocks.append(tf.pad(
        tensor=matrix,
        paddings=tf.concat(
            [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
             [(row_before_length, row_after_length)]],
            axis=0)))
  blocked = tf.concat(row_blocks, -2)
  blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
  return blocked

def h5read(filename, datasets):
    if type(datasets) != list:
        datasets = [datasets]
    with h5py.File(filename, 'r') as h5f:
        data = [h5f[ds][:] for ds in datasets]
    return data

class i_vec_layer(Layer):
    def __init__(self,tv_dim, ndim, nmix,ubmFilename,T_mat_filename,**kwargs):
        super(i_vec_layer, self).__init__(**kwargs)
        self.tv_dim = tv_dim
        self.ndim = ndim
        self.nmix = nmix
        self.itril = np.tril_indices(tv_dim)
        self.Sigma = np.empty((self.ndim * self.nmix, 1))
        self.T_iS = None  # np.empty((self.tv_dim, self.ndim * self.nmix))
        self.T_iS_Tt = None  # np.empty((self.nmix, self.tv_dim * (self.tv_dim+1)/2))
        self.Tm = np.empty((self.tv_dim, self.ndim * self.nmix))
        self.Im = np.eye(self.tv_dim,dtype='float32')
        self.ubmFilename=ubmFilename;
        self.T_mat_filename=T_mat_filename;
        self.mu, self.sigma, self.w = h5read(ubmFilename, ['means', 'variances', 'weights'])
        self.Sigma = self.sigma.reshape((1, self.ndim * self.nmix), order='F')
        self.tmat_init()
        #self.T_iS_Tt = self.comp_T_invS_Tt()
        self.C_ = self.compute_C();
        self.T_iS=self.T_iS.astype('float32')
        self.Tm=self.Tm.astype('float32')
        L = np.ones((self.tv_dim, self.tv_dim))
        self.mask_mat=np.tril(L, -1);
	#print('shape_Tis:'+str(self.Tm.shape))
	#print('shape_Sigma:'+str(self.sigma.shape))
	#print(self.T_iS)

    def tmat_init(self):
        self.Tm = h5read(self.T_mat_filename, 'T')[0]	
        self.T_iS = self.Tm / self.Sigma

    def build(self, input_shape):
        super(i_vec_layer, self).build(input_shape)  # Be sure to call this at the end

    def call(self,x):
        N,F_h=self.compute_centered_stats(tf.transpose(x));
        N_c=tf.tile(N,[self.ndim,1])
        N_c=tf.reshape(tf.transpose(N_c),[-1])
        N_u=tf.diag(N_c)
        term0=tf.matmul(self.T_iS,N_u);
        term1=tf.matmul(term0,self.Tm,transpose_b=True)
        Mat_inv=tf.matrix_inverse(self.Im+term1);
        term2=tf.matmul(Mat_inv,self.T_iS);
        i_vec=tf.matmul(term2,F_h)	
        otherinfo=[N,F_h,N_c,term0,term1,term2,Mat_inv];
        return(i_vec);

    def postprob(self, data):
        post = self.lgmmprob(data)
        llk = logsumexp(post, 0)
        post = tf.exp(post - llk)
        return post, llk

    def compute_C(self):
        precision = 1 / self.sigma
        log_det = tf.reduce_sum(tf.log(self.sigma), axis=0, keepdims=True)
        prod=self.mu*self.mu*precision;
        prod_sum=tf.reduce_sum(prod, axis=0, keepdims=True);
        return prod_sum + log_det - 2*tf.log(self.w)

    def lgmmprob(self, data):
        precision = 1/self.sigma
        term1=tf.matmul(precision,tf.multiply(data,data),transpose_a=True);
        term2=2*tf.matmul(self.mu*precision,data,transpose_a=True);
        D = term1-term2 + self.ndim * tf.log(2*np.pi)
        return -0.5 * (tf.transpose(self.C_) + D)

    @staticmethod
    def compute_zeroStat(post):
        return tf.transpose(tf.reduce_sum(post, axis=1, keepdims=True))

    @staticmethod
    def compute_firstStat(data, post):
        return tf.matmul(data,post,transpose_b=True)

    @staticmethod
    def compute_llk(post):
        return logsumexp(post, 1)

    def compute_centered_stats(self, data):
        post = self.postprob(data)[0]
        N = self.compute_zeroStat(post)
        F = self.compute_firstStat(data, post)
        F_hat = tf.reshape(tf.transpose(F - self.mu * N), (self.ndim * self.nmix, 1))
        return N, F_hat

    def compute_output_shape(self, input_shape):
        return (self.tv_dim,1)

def h5read(filename, datasets):
    if type(datasets) != list:
        datasets = [datasets]
    with h5py.File(filename, 'r') as h5f:
        data = [h5f[ds][:] for ds in datasets]
    return data

def logsumexp(x, dim):
    xmax = tf.reduce_max(x,axis=dim, keepdims=True)
    y = xmax + tf.log(tf.reduce_sum(tf.exp(x-xmax), axis=dim, keepdims=True))
    return y
