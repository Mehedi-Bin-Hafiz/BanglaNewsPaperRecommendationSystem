#coding = utf - 8
import joblib
import pandas as pd
model = joblib.load('C:\\Users\\Mehedi\\PycharmProjects\\NewsPaperSavedModel\\WithOutStopSGDbestMLModel')


df = pd.read_excel('../database/test_dataset.xlsx')
predict_list = list()
for i in df.News:
    predict_list.append(model.predict([i]))
df['prediction_target'] = predict_list

df.to_excel('../database/prediction_dataset.xlsx')

