import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('../database/prediction_dataset.xlsx',engine= 'openpyxl')

test_df = df.Target.value_counts().rename_axis('target').reset_index(name='frequency')
test_df = test_df.sort_values(by =['target'])
pred_df = df.prediction_target.value_counts().rename_axis('target').reset_index(name='frequency')
pred_df = pred_df.sort_values(by = ['target'])

print(test_df)
print(pred_df)

test_fre = test_df.frequency.to_list()
pred_fre = pred_df.frequency.to_list()
print(test_fre)
print(pred_fre)
targets = pred_df.target.to_list()

RealClass = test_fre
Predictedclass = pred_fre
labels = targets
x = np.arange(len(labels))

width=0.25
fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, RealClass, width, label='Real')
rects2 = ax.bar(x + width/2, Predictedclass, width, label='Predict')
ax.set_xticks(x)
ax.set_xticklabels(labels,  rotation='vertical')
ax.legend()
plt.grid()

plt.savefig('realvspredictedsentiment.png') # need to call before calling show
plt.show()