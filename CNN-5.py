
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
print(tf.config.experimental.list_physical_devices('GPU'))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def block(inputs):
    tmp=keras.layers.Conv2D(16,(3,3),1,'same')(inputs) #(32,32,16)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp=tf.nn.relu(tmp)
    tmp=keras.layers.MaxPooling2D((2,2),2)(tmp)
    tmp=keras.layers.Conv2D(32,(3,3),1,'same')(tmp) #(16,16,32)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp=tf.nn.relu(tmp)
    tmp=keras.layers.MaxPooling2D((2,2),2)(tmp)
    tmp=keras.layers.Conv2D(64,(3,3),1,'same')(tmp) #(8,8,64)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp=tf.nn.relu(tmp)
    tmp=keras.layers.MaxPooling2D((2,2),2)(tmp)
    tmp=keras.layers.Conv2D(64,(3,3),1,'same')(tmp) #(4,4,64)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp=tf.nn.relu(tmp)
    tmp=keras.layers.MaxPooling2D((2,2),2)(tmp)
    tmp=keras.layers.Conv2D(64,(3,3),1,'same')(tmp) #(2,2,64)
    tmp=keras.layers.BatchNormalization()(tmp)
    tmp=tf.nn.relu(tmp)
    tmp=keras.layers.MaxPooling2D((2,2),2)(tmp)
    return tmp #(2,64)
def classifer(inputs):
    tmp=keras.layers.GlobalAveragePooling2D()(inputs)
    tmp=keras.layers.Dropout(0.2)(tmp)
    tmp=keras.layers.Dense(64)(tmp)
    tmp=keras.layers.Activation('relu')(tmp)
    tmp=keras.layers.Dropout(0.2)(tmp)
    tmp=keras.layers.Dense(32)(tmp)
    tmp=keras.layers.Activation('relu')(tmp)
    tmp=keras.layers.Dense(10)(tmp)
    return tmp
def CNN_5():
    inputs=keras.Input(shape=(32,32,1))
    tmp=block(inputs)
    tmp=classifer(tmp)
    return keras.Model(inputs=inputs,outputs=tmp)

model=CNN_5()
keras.utils.plot_model(model, "CNN_5.png",show_shapes=True)

import numpy as np
x_s=np.load('oral_data/D_data.npy')
x_t=np.load('oral_data/C_data.npy')
y_s=np.load('oral_data/D_label.npy')
y_t=np.load('oral_data/C_label.npy')

model.compile(optimizer='adam',loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['acc'])

model.fit(x=np.reshape(x_s,newshape=(-1,32,32,1)),y=y_s,batch_size=64,epochs=200,verbose=1,
          validation_data=(np.reshape(x_t,newshape=(-1,32,32,1)),y_t))