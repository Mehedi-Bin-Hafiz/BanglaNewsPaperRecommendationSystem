#coding = utf - 8
import joblib
import pandas as pd
model = joblib.load('C:\\Users\\Mehedi\\PycharmProjects\\NewsPaperSavedModel\\WithOutStopSGDbestMLModel')


df = pd.read_excel('../database/test_dataset.xlsx')
predict_list = list()
for num, i in enumerate(df.News):
    predict_list.append(model.predict([i])[0])
    print(num, " is completed")


df['prediction_target'] = predict_list

df.to_excel('../database/prediction_dataset.xlsx',index= False)

