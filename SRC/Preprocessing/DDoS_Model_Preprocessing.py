import pandas as pd
import os

data=pd.read_csv("Data/Raw/cicddos2019_dataset.csv")

data1=data.drop_duplicates()

data=data1.dropna(subset=['Class'])

dataX=pd.get_dummies(data.drop(columns=['Class','Label']))
dataY=data['Class']

processed_folder ="Data/Processed"

os.makedirs(processed_folder,exist_ok=True)

dataX.to_csv(os.path.join(processed_folder,"DDoS-Model-Processed-X.csv"),index=False)
dataY.to_csv(os.path.join(processed_folder,"DDoS-Model-Processed-Y.csv"),index=False)