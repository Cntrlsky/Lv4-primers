import pandas as pd
from SRC.Feature_Engineering.DDoS_Feature_Engineering import DDoS_Feature_Engineering as DDoS

data=pd.read_csv("Data/Raw/cicddos2019_dataset.csv")

data1=data.drop_duplicates()

data=data1.dropna(subset=['Class'])

X=DDoS(data)
Y=data['Class']

X.to_csv("Data/Processed/DDoS_Model_Processed_X.csv",index=False)
Y.to_csv("Data/Processed/DDoS_Model_Processed_Y.csv",index=False)
