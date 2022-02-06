import joblib
import pandas as pd
model = joblib.load('E:\\SavedModel\\NewsPaperSavedModel\\WithOutStopSGDbestMLModel')

en_df = pd.read_excel('../database/EngineDataset.xlsx',engine='openpyxl')

user_in = input('Enter News Article: ')
pred = model.predict([user_in])[0]

selected_row = en_df[en_df.target == pred]

print('Prediction target: ',pred)
print('Suggested headline: ',selected_row.headline)
print('Suggested next url ',selected_row.url)