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
from Model1 import GetModel
os.environ["CUDA_VISIBLE_DEVICES"]="0"


root='./NewFeats/'; # define root directory
file_list='all_train_files';
expts_dir='../Models/'


def cos_distance(y_true, y_pred):
    y_true=tf.transpose(y_true)
    y_true = tf.nn.l2_normalize(y_true,axis=0)
    y_pred = tf.nn.l2_normalize(y_pred,axis=0)
    return 1.0-K.sum(tf.multiply(y_true,y_pred))



class DataGenerator(Sequence):
    def __init__(self,filename,tr_flg):
        files=open(filename,'r');
        self.files=files.readlines();
        self.files=[x.strip() for x in self.files];
        if(tr_flg==0):
            self.files=self.files[12883:]
        else:
            self.files=self.files[0:12883]
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        #print(self.files[idx])
        name=self.files[idx]
        parts=name.split('/');
        if(parts[1] in 'wTIMIT'):
            Y=load_h5py(root+'ivectors/'+name[2:],'ivec');
            name=name.replace('normal','whisper')
            name=name.replace('n.h5','w.h5')
            X=load_h5py(root+'features/'+name[2:],'fea');
        else:
            Y=load_h5py(root+'ivectors/'+name[2:],'ivec');
            name=name.replace('Neutral','Whisper')
            name=name.replace('/n','/w')
            X=load_h5py(root+'features/'+name[2:],'fea');

        X=X.transpose();
        
        X=X[np.newaxis,:,:];
        Y=Y.transpose()
        #print([X.shape,Y.shape])
        return(X,Y)

def load_h5py(filename,key):
    f=h5py.File(filename,'r')
    #print(f.keys())
    return(f[key].value)	
	
F=39;	
ivec_dim=400;
nmin=512


## model buidling
start=time.time();

inp,out=GetModel()

ivec_layer=i_vec_layer(ivec_dim, F, nmin,root+'ubm.gmm',root+'tvmat.h5');
ivec_out=Lambda(lambda x:ivec_layer.call(x[0,:,:]),output_shape=(400,))(out)
model=Model(inp,ivec_out);
model.compile(optimizer='adam',loss=cos_distance,metrics=['mse']);
model.summary();

callback1=keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=8,verbose=0, mode='auto')
callback2=keras.callbacks.ModelCheckpoint(expts_dir+'best_model2', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
print('..fit model')	
history=model.fit_generator(DataGenerator(file_list,1),epochs=50,validation_data=DataGenerator(file_list,0),callbacks=[callback1,callback2],workers=1, use_multiprocessing=False)
	

