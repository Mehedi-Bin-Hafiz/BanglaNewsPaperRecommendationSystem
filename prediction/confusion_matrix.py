import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import seaborn as sns
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 10})

df = pd.read_excel(r'../database/prediction_dataset.xlsx',engine='openpyxl')


for value in set(df.Target.tolist()):
    tempdf = df.loc[df.Target == value]
    print('original target {}'.format(value),len(tempdf))
    real = tempdf['Target'].values
    predicted = tempdf['prediction_target'].values
    cf_matrix = confusion_matrix(real,predicted)
    print(cf_matrix)