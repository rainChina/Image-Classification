import keras.backend as k
import numpy as np
from keras.utils import np_utils
import pandas as pd
from load_data import loadData
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras.layers import Dropout
import os

k.set_image_dim_ordering('tf')
seed = 122
np.random.seed(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def pre_process(x):
    x = x.astype('float32')
    x = x/255.0
    return x

def one_hot_encode(y):
    y = LabelEncoder().fit_transform(y)
    # y = OneHotEncoder().fit_transform(y)
    y = np_utils.to_categorical(y)
    print(y)
    print('The shape of y',y.shape)
    num = y.shape[1]
    return y, num

def define_LeNet5(class_number):
    model = Sequential()
    model.add(Conv2D(filters=6,kernel_size=5,strides=1,padding='valid',input_shape=(32,32,1),activation='relu'))
    model.add(MaxPool2D(pool_size=2,strides=2))
    model.add(Dropout(0.6))
    model.add(Conv2D(filters=16,kernel_size=5,strides=1,padding='valid',activation='relu'))
    model.add(MaxPool2D(pool_size=2,strides=2))
    model.add(Dropout(0.6))
    model.add(Flatten())
    model.add(Dense(units=128,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units=64,activation='relu'))
    model.add(Dropout(0.4))
    # model.add(Dense(class_number,activation='softmax'))
    # model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    model.add(Dense(class_number, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

X,Y = loadData()

X = pre_process(X)

Y,num = one_hot_encode(Y)

print(num)
# print(type(Y))
# print(Y.shape)
# print(Y)

X_train,  X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=122)



model = define_LeNet5(num)
# model.load_weights("model_22_dropout.h5")
# print('X_train shape:',X_train.shape)
# print('Y_train shape:',Y_train.shape)

tbCallBack = TensorBoard(log_dir='./Graph',histogram_freq=0,write_graph=True,write_images=False)
mc = ModelCheckpoint('./mode_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_weights_only=True, period=66)
model.fit(X_train,Y_train,batch_size=64,epochs=1222,validation_data=(X_test, Y_test),
          shuffle=True, callbacks=[tbCallBack,mc])

score = model.evaluate(X_test,Y_test)
print(score)
print('score %1.2f%%'%(100*score[1]))

# model.fit(X_train,Y_train,batch_size=32,epochs=2)
#
# # model.to_json('model_122')
#
model.save('model_22_dropout.h5')

# model.load_weights('model_22.h5')
# score = model.evaluate(X_test,Y_test)
# print(score)
# print('score %1.2f%%'%(100*score[1]))