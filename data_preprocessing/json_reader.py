# import pandas as pd
# import json
#
# with open('/content/drive/MyDrive/Dataset - 01/pro-part-4.json') as json_file:
#     data = json_file.readlines()
#     # this line below may take at least 8-10 minutes of processing for 4-5 million rows. It converts all strings in list to actual json object
#     data = list(map(json.loads, data))
# df = pd.json_normalize(data[0])
# df.to_excel('/content/drive/MyDrive/colab/raw_dataset/raw_dataset_1.xlsx',index=False)