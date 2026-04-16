import pandas as pd
import os

data=pd.read_csv("Data/Raw/embedded_system_network_security_dataset.csv")

data01=data.drop_duplicates()

data02=data01.dropna(subset=['label'])

X=pd.get_dummies(data02.drop(columns=['label']))
Y=data02['label']

X.to_csv("Data/Processed/Network_AnomalyX.csv",index=False)
Y.to_csv("Data/Processed/Network_AnomalyY.csv",index=False)