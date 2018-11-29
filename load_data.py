import os
import pandas as pd
import cv2
import numpy as np

def loadData():
    # get the name-label dictionary
    labelFrm = pd.read_csv('afm_clean.csv')
    labelDic = dict([(lb, nm) for lb, nm in zip(labelFrm.imName,labelFrm.noise_simple)])
    # print(labelDic)

    X=[]
    Y=[]
    root = 'data/images/'
    names = os.listdir(root)
    for name in names:
        path = os.path.join(root,name)
        # print(path)
        img = cv2.imread(path)
        img = cv2.resize(img,(32,32))[0]
        X.append(img)
        Y.append(labelDic[name])

        # print(type(X))
        #
        # X = np.array(X).astype('float32')
        #
        # print(type(img))
        # print(img.shape)
        # print(k.image_dim_ordering(img))
        # print(img)

        # break

    return np.array(X), np.array(Y)

loadData()