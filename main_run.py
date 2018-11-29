import keras.backend as k
import numpy as np
from keras.utils import np_utils
import pandas as pd
from load_data import loadData
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

k.set_image_dim_ordering('tf')
seed = 122
np.random.seed(seed)

def pre_process(x):
    x = x.astype('float32')
    x = x/255.0
    return x

def one_hot_encode(y):
    y = np_utils.to_categorical(y)
    num = y.shape[1]
    return y, num

def define_LeNet5(class_number):
    model = Sequential()
    model.add(Conv2D(filters=6,kernel_size=5,strides=1,padding='valid',input_shape=(32,32,1),activation='relu'))
    model.add(MaxPool2D(pool_size=2,strides=2))
    model.add(Conv2D(filters=16,kernel_size=5,strides=1,padding='valid',activation='relu'))
    model.add(MaxPool2D(pool_size=2,strides=2))
    model.add(Flatten())
    model.add(Dense(units=120,activation='relu'))
    model.add(Dense(units=84,activation='relu'))
    model.add(Dense(units=class_number,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    print(model.summary())
    return model

X,Y = loadData()

X = pre_process(X)

Y,num = one_hot_encode(Y)

X_train, Y_train, X_test, Y_test = train_test_split(X,Y, test_size=0.2, random_state=22)

model = define_LeNet5(num)

model.fit(X_train,Y_train,batch_size=64,epochs=2)

model.to_json('model_122')
