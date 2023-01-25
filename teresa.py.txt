import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Flatten,Conv2D, DepthwiseConv2D, Input, Add, AveragePooling2D,Dense
from tensorflow.keras.optimizers import Adam,Nadam,SGD
from tensorflow.keras.losses import CategoricalCrossentropy,SparseCategoricalCrossentropy

num_classes = 10  
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
img_height, img_width, channel = X_train.shape[1],X_train.shape[2],X_train.shape[3]
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

print('Shape of X_train  : ', X_train.shape)
print('Shape of X_test : ', X_test.shape)
print('Shape of y_train : ', y_train.shape)
print('Shape of y_test : ', y_test.shape)


X_train = X_train/255.0
X_test = X_test/255.0

def BottleNeck(input, t, c, s):
  inp_filters = input.shape[-1]
  if s == 1:
    conv_1 = Conv2D(filters = inp_filters * t, kernel_size = 1)(input)  
    BN = BatchNormalization()(conv_1)
    Relu6 = tf.keras.activations.relu(BN, max_value=6)
    
    Dwise_3 = DepthwiseConv2D(kernel_size=3, padding = 'same')(Relu6)
    BN = BatchNormalization()(Dwise_3)
    Relu6 = tf.keras.activations.relu(BN, max_value=6)

    conv_2 = Conv2D(filters = c, kernel_size = 1)(Relu6)  
    BN_out = BatchNormalization()(conv_2)

    if inp_filters != c:
        reduced = Conv2D(filters = c, kernel_size = 1)
        output = Add()([reduced(input), BN_out])
    else:
        output = Add()([input, BN_out])

    return output
    
  else:
    conv_1 = Conv2D(filters = inp_filters * t, kernel_size = 1)(input)  
    BN = BatchNormalization()(conv_1)
    Relu6 = tf.keras.activations.relu(BN, max_value=6)

    Dwise_3 = DepthwiseConv2D(kernel_size=3, padding = 'same', strides = s)(Relu6)
    BN = BatchNormalization()(Dwise_3)
    Relu6 = tf.keras.activations.relu(BN, max_value=6)

    conv_2 = Conv2D(filters = c, kernel_size = 1)(Relu6)  
    BN_out = BatchNormalization()(conv_2)

    return BN_out


tf.keras.backend.clear_session
tf.random.set_seed(16)
input = Input(shape=(img_height, img_width, channel,))

conv = Conv2D(filters = 32,activation='relu', kernel_size=3, strides=2)(input)
BN = BatchNormalization()(conv)
b = BottleNeck(BN, t = 1, c = 16, s = 1)

b = BottleNeck(b, t = 6, c = 24, s = 2)
b = BottleNeck(b, t = 6, c = 24, s = 2)

b = BottleNeck(b, t = 6, c = 32, s = 2)
b = BottleNeck(b, t = 6, c = 32, s = 2)
b = BottleNeck(b, t = 6, c = 32, s = 2)

b = BottleNeck(b, t = 6, c = 64, s = 2)
b = BottleNeck(b, t = 6, c = 64, s = 2)
b = BottleNeck(b, t = 6, c = 64, s = 2)
b = BottleNeck(b, t = 6, c = 64, s = 2)

b = BottleNeck(b, t = 6, c = 96, s = 1)
b = BottleNeck(b, t = 6, c = 96, s = 1)
b = BottleNeck(b, t = 6, c = 96, s = 1)

b = BottleNeck(b, t = 6, c = 160, s = 2)
b = BottleNeck(b, t = 6, c = 160, s = 2)
b = BottleNeck(b, t = 6, c = 160, s = 2)

b = BottleNeck(b, t = 6, c = 320, s = 1)

conv = Conv2D(filters = 1280,activation='relu', kernel_size=1)(b)
BN = BatchNormalization()(conv)

avgpool = AveragePooling2D(pool_size=(7, 7), padding='same')(BN)

conv = Conv2D(filters = 10, kernel_size=1)(avgpool)
BN = BatchNormalization()(conv)

output = Flatten()(BN)
output = Dense(128)(output)
output = Dense(num_classes, activation='softmax')(output)
model = Model(inputs=[input], outputs=[output])

model.compile(loss=CategoricalCrossentropy(),
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])


history= model.fit(X_train,y_train,batch_size=128,epochs=100,verbose=1)
