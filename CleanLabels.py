import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



def afm_clean_export(src, aim):
    NUM_VALID = 3360
    dataFrm = pd.read_csv(src)
    data = dataFrm.loc[0:NUM_VALID]
    data = data.drop(['secondaryNoise','fiber'], axis=1)

    # print(type(data['noise'].value_counts().plot))
    # data['noise'].value_counts().plot.pie(autopct='%1.1f%%')
    # data['noise'].value_counts().plot.bar()

    data.loc[:,'noise'].fillna(value='c', inplace=True)

    noiseTypes = []
    for item in ['hs', 'hg', 'hp', 'vg', 'hl', 'vl', 'hb']:
        data['noise'] = data['noise'].str.replace(item,item[1:])

    cnt = 0;
    for v in data['imPath']:
        data.loc[cnt,'imName'] = os.path.basename(v)
        cnt += 1

    data['noise_simple'] = data['noise'].copy()
    data.loc[data['noise_simple']!='c', 'noise_simple'] = 'n'

    data.to_csv(aim,index=False)
    # plt.subplot(2,1,1)
    # data['noise_simple'].value_counts().plot.bar()
    # plt.subplot(2,1,2)
    # data['noise_simple'].value_counts().plot.pie(autopct='%1.1f%%')
    # plt.show()

    # {nan,  'hs',  'hg', 'hp',  'vg', 'w', 'hl', 'vl', 'p', 'n', 'x', 'hb'}

def split(src, train_file, test_file):
    dfRaw = pd.read_csv(src)
    dfTrain, dfTest = train_test_split(dfRaw, test_size=0.3, random_state=122)
    # plt.subplot(2,1,1)
    # dfTrain['noise'].value_counts().plot.bar()
    # plt.subplot(2,1,2)
    # dfTest['noise'].value_counts().plot.bar()
    # plt.show()
    # print(dfTrain.head())
    # dfShuffle = dfRaw.sample(frac=1).reset_index(drop=True)
    dfTrain[['id','noise_simple']].to_csv(train_file,index=False)
    dfTest[['id','noise_simple']].to_csv(test_file,index=False)

# afm_clean_export('afm.csv','afm_clean.csv')

split('afm_clean.csv','afm_train.csv', 'afm_test.csv')