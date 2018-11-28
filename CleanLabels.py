import pandas as pd
import os
import matplotlib.pyplot as plt

NUM_VALID = 3360
dataFrm = pd.read_csv('afm.csv')
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
    # if(cnt>6):
    #     break

# data.to_csv('afm_clean.csv',index=False)
plt.subplot(2,1,1)
data['noise'].value_counts().plot.bar()
plt.subplot(2,1,2)
data['noise'].value_counts().plot.pie(autopct='%1.1f%%')
plt.show()

# {nan,  'hs',  'hg', 'hp',  'vg', 'w', 'hl', 'vl', 'p', 'n', 'x', 'hb'}