import pandas as pd
import os

data=pd.read_csv("Data/Raw/cicddos2019_dataset.csv")

data1=data.drop_duplicates()

data=data1.dropna(subset=['Class'])

X=pd.get_dummies(data.drop(columns=['Class','Label']))
Y=data['Class']

processed_folder ="Data/Processed"

os.makedirs(processed_folder,exist_ok=True)

X.to_csv(os.path.join(processed_folder,"DDoS_Model_Processed_X.csv"),index=False)
Y.to_csv(os.path.join(processed_folder,"DDoS_Model_Processed_Y.csv"),index=False)