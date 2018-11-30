import os
import sys
import pandas as pd
import cv2
import numpy as np

def loadData():
    print('start to load data...')
    # get the name-label dictionary
    labelFrm = pd.read_csv('afm_clean.csv')
    labelDic = dict([(lb, nm) for lb, nm in zip(labelFrm.imName,labelFrm.noise_simple)])
    # print(labelDic)

    X=[]
    Y=[]
    root = 'data/images/'
    names = os.listdir(root)
    fCnt = len(names)
    cnt = 1
    for name in names:
        show_state_percentage(cnt,fCnt)
        path = os.path.join(root,name)
        # print(path)
        img = cv2.imread(path)
        img = cv2.resize(img,(32,32))[:,:,0]
        imgN = img[:,:,np.newaxis]
        X.append(imgN)
        Y.append(labelDic[name])
        cnt+=1

        # print(type(X))
        #
        # X = np.array(X).astype('float32')
        #
        # print(type(img))
        # print(img.shape)
        # print(k.image_dim_ordering(img))
        # print(img)

        # break
    X = np.array(X)
    Y = np.array(Y)
    print('\r')
    print('loading completed!')
    print('X shape:',X.shape)
    print('Y shape:',Y.shape)
    return X, Y

def show_state_percentage(num, count):
    if(num%10==0):
        sys.stdout.write('\r')
        print('loading...', '%1.2f%%'%(float(num)/float(count)), end='', flush=True)

# loadData()