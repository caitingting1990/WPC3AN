import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
print(tf.config.experimental.list_physical_devices('GPU'))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def block0(inputs):#(32,32,1)
    tmp=keras.layers.Conv2D(32,(7,7),2,'same')(inputs)
    tmp=keras.layers.MaxPooling2D((2,2),2,'same')(tmp)#(8,8,32)
    return tmp
def block1(inputs):
    tmp=keras.layers.Conv2D(32,(3,3),1,'same')(inputs)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp=tf.nn.relu(tmp)
    tmp=keras.layers.Conv2D(32,(3,3),1,'same')(tmp)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp1=tf.nn.relu(inputs)+tmp #(8,8,32)
    tmp=keras.layers.Conv2D(32,(3,3),1,'same')(tmp1)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp=tf.nn.relu(tmp)
    tmp=keras.layers.Conv2D(32,(3,3),1,'same')(tmp)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp=tf.nn.relu(tmp1)+tmp #(8,8,32)
    return tmp
def block2(inputs):
    tmp=keras.layers.Conv2D(64,(3,3),2,'same')(inputs) #(4,4,64)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp=tf.nn.relu(tmp)
    tmp=keras.layers.Conv2D(64,(3,3),1,'same')(tmp)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp0=keras.layers.Conv2D(64,(1,1),2,'same')(inputs)
    tmp1=tf.nn.relu(tmp0)+tmp  #(4,4,64)
    tmp=keras.layers.Conv2D(64,(3,3),1,'same')(tmp1)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp=tf.nn.relu(tmp)
    tmp=keras.layers.Conv2D(64,(3,3),1,'same')(tmp)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp=tf.nn.relu(tmp)
    return tf.nn.relu(tmp1)+tmp #(4,4,64)
def block3(inputs):
    tmp=keras.layers.Conv2D(128,(3,3),2,'same')(inputs) #(2,2,128)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp=tf.nn.relu(tmp)
    tmp=keras.layers.Conv2D(128,(3,3),1,'same')(tmp)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp0=keras.layers.Conv2D(128,(1,1),2,'same')(inputs)
    tmp1=tf.nn.relu(tmp0)+tmp  #(2,2,256)
    tmp=keras.layers.Conv2D(128,(3,3),1,'same')(tmp1)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp=tf.nn.relu(tmp)
    tmp=keras.layers.Conv2D(128,(3,3),1,'same')(tmp)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp=tf.nn.relu(tmp)
    return tf.nn.relu(tmp1)+tmp #(2,2,128)
def block4(inputs):
    tmp=keras.layers.Conv2D(256,(3,3),1,'same')(inputs) #(2,2,256)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp=tf.nn.relu(tmp)
    tmp=keras.layers.Conv2D(256,(3,3),1,'same')(tmp)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp0=keras.layers.Conv2D(256,(1,1),1,'same')(inputs)
    tmp1=tf.nn.relu(tmp0)+tmp  #(2,2,256)
    tmp=keras.layers.Conv2D(256,(3,3),1,'same')(tmp1)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp=tf.nn.relu(tmp)
    tmp=keras.layers.Conv2D(256,(3,3),1,'same')(tmp)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp=tf.nn.relu(tmp)
    return tf.nn.relu(tmp1)+tmp #(2,2,256)
def classifer(inputs):
    tmp=keras.layers.GlobalAveragePooling2D()(inputs)
    tmp=keras.layers.Dropout(0.2)(tmp)
    tmp=keras.layers.Dense(128)(tmp)
    tmp=keras.layers.Activation('relu')(tmp)
    tmp=keras.layers.Dropout(0.2)(tmp)
    tmp=keras.layers.Dense(64)(tmp)
    tmp=keras.layers.Activation('relu')(tmp)
    tmp=keras.layers.Dense(10)(tmp)
    return tmp
def ResNet18():
    inputs=keras.Input(shape=(32,32,1))
    tmp=block0(inputs)
    tmp=block1(tmp)
    tmp=block2(tmp)
    tmp=block3(tmp)
    tmp=block4(tmp)
    tmp=classifer(tmp)
    return keras.Model(inputs=inputs,outputs=tmp)

model=ResNet18()

import numpy as np
x_s=np.load('oral_data/D_data.npy')
x_t=np.load('oral_data/C_data.npy')
y_s=np.load('oral_data/D_label.npy')
y_t=np.load('oral_data/C_label.npy')

model.compile(optimizer='adam',loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['acc'])

model.fit(x=np.reshape(x_s,newshape=(-1,32,32,1)),y=y_s,batch_size=64,epochs=200,verbose=1,validation_data=(np.reshape(x_t,newshape=(-1,32,32,1)),y_t))
