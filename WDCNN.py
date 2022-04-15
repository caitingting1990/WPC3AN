import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
print(tf.config.experimental.list_physical_devices('GPU'))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def block(inputs):
    tmp=keras.layers.Conv1D(16,(64,),8,'same')(inputs) #(128,16)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp=tf.nn.relu(tmp)
    tmp=keras.layers.MaxPooling1D((2,),2)(tmp)
    tmp=keras.layers.Conv1D(32,(3,),1,'same')(tmp) #(32,32)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp=tf.nn.relu(tmp)
    tmp=keras.layers.MaxPooling1D((2,),2)(tmp)
    tmp=keras.layers.Conv1D(64,(3,),1,'same')(tmp) #(16,64)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp=tf.nn.relu(tmp)
    tmp=keras.layers.MaxPooling1D((2,),2)(tmp)
    tmp=keras.layers.Conv1D(64,(3,),1,'same')(tmp) #(8,64)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp=tf.nn.relu(tmp)
    tmp=keras.layers.MaxPooling1D((2,),2)(tmp)    #(4,64)
    tmp=keras.layers.Conv1D(64,(3,),1,'same')(tmp) #(4,64)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp=tf.nn.relu(tmp)
    tmp=keras.layers.MaxPooling1D((2,),2)(tmp)    #(2,64)
    return tmp #(2,64)
def classifer(inputs):
    tmp=keras.layers.GlobalAveragePooling1D()(inputs)
    tmp=keras.layers.Dropout(0.2)(tmp)
    tmp=keras.layers.Dense(64)(tmp)
    tmp=keras.layers.Activation('relu')(tmp)
    tmp=keras.layers.Dropout(0.2)(tmp)
    tmp=keras.layers.Dense(32)(tmp)
    tmp=keras.layers.Activation('relu')(tmp)
    tmp=keras.layers.Dense(10)(tmp)
    return tmp
def WDCNN():
    inputs=keras.Input(shape=(1024,1))
    tmp=block(inputs)
    tmp=classifer(tmp)
    return keras.Model(inputs=inputs,outputs=tmp)

model=WDCNN()
keras.utils.plot_model(model, "WDCNN.png",show_shapes=True)

import numpy as np
x_s=np.load('oral_data/D_data.npy')
x_t=np.load('oral_data/C_data.npy')
y_s=np.load('oral_data/D_label.npy')
y_t=np.load('oral_data/C_label.npy')

model.compile(optimizer='adam',loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['acc'])

model.fit(x=np.reshape(x_s,newshape=(-1,1024,1)),y=y_s,batch_size=64,epochs=200,verbose=1,
          validation_data=(np.reshape(x_t,newshape=(-1,1024,1)),y_t))
