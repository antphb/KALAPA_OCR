import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional, Add, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, sigmoid, softmax
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from config import Config


class CRNN:
    def __init__(self,input_shape, num_classes,max_label_len, filters= Config.filters, pool_sizes=Config.pool_sizes, strides=Config.strides, lstm_units=Config.lstm_units, dropout_rate=0.2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.max_label_len= max_label_len
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        
        self.filters = filters
        self.pool_sizes=pool_sizes
        self.strides=strides
        self.crnn_model = None
        self._build_model()
    
    def _conv_block(self,x,i,MaxPool=False,BatchNormal=False,last_block=False):
        i=i-1
        x=Conv2D(self.filters[i],(3,3),padding='same')(x)
        if BatchNormal:
            x=BatchNormalization()(x)
        if MaxPool and last_block:
            x=MaxPool2D(pool_size=self.pool_sizes[i])(x)
        elif MaxPool:
            x=MaxPool2D(pool_size=self.pool_sizes[i],strides=self.strides[i])(x)

        x=Activation('relu')(x)
        return x

    def _residual_block(self,x,i):
        i=i-1
        y=Conv2D(self.filters[i],(3,3),padding='same')(x)
        y=BatchNormalization()(y)
        y=Add()([y,x])
        y=Activation('relu')(y)
        return y
    
    def ctc_lambda_func(self,args):
        y_pred, labels, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    
    def _build_model(self):
        inputs=Input(shape=self.input_shape)
        # Block 1
        x=self._conv_block(inputs,1,MaxPool=True)
        # print("Block 1",x.shape)
        # x_1=x
        # Block 2
        x=self._conv_block(x,2,MaxPool=True)
        # print("Block 2",x.shape)
        # Block 3
        x=self._conv_block(x,3,BatchNormal=True)
        # print("Block 3",x.shape)
        # Block 4
        x=self._residual_block(x,4)
        # print("Block 4",x.shape)
        # Block 5
        x=self._conv_block(x,5,BatchNormal=True)
        # print("Block 5",x.shape)
        # Block 6
        x=self._residual_block(x,6)
        # print("Block 6",x.shape)
        # Block 7
        x=self._conv_block(x,7,BatchNormal=True,MaxPool=True,last_block=True)
        # print("Block 7",x.shape)
        
        x = MaxPool2D(pool_size=(3,1))(x)
        # print(x.shape)
        
        squeezed = Lambda(lambda x: K.squeeze(x, 1))(x)
        
        blstm_1 = Bidirectional(LSTM(self.lstm_units, return_sequences=True, dropout = self.dropout_rate))(squeezed)
        blstm_2 = Bidirectional(LSTM(self.lstm_units, return_sequences=True, dropout = self.dropout_rate))(blstm_1)
        
        outputs = Dense(self.num_classes, activation = 'softmax')(blstm_2)
        
        self.crnn_model = Model(inputs, outputs)
        
    def summary(self):
        self.crnn_model.summary()
        
    def get_model(self):
        return self.crnn_model
    
    def compile(self):
        labels = Input(name='the_labels', shape=[self.max_label_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        
        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([self.crnn_model.output, labels, input_length, label_length])
        
        self.model = Model(inputs=[self.crnn_model.input, labels, input_length, label_length], outputs=loss_out)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        
        self.model.compile(loss = {'ctc': lambda y_true, y_pred: y_pred}, optimizer = optimizer)
        return self.model
    
    def load_weights(self, path):
        self.crnn_model.load_weights(path)
        
    def predict(self, input_batch):
        return self.crnn_model.predict(input_batch)