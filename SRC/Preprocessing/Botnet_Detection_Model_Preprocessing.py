import pandas as pd
from SRC.Feature_Engineering.Botnet_Feature_Engineering import Botnet_Feature_Engineering

data01=pd.read_csv("Data/Raw/CTU13_Attack_Traffic.csv")
data11=pd.read_csv("Data/Raw/CTU13_Normal_Traffic.csv")

data02=data01.drop_duplicates()
data12=data11.drop_duplicates()

data03=data02.dropna(subset=['Label'])
data13=data12.dropna(subset=['Label'])

data=pd.concat([data03,data13])
X=Botnet_Feature_Engineering(data)
Y=data['Label']

X.to_csv("Data/Processed/Botnet_X.csv",index=False)
Y.to_csv("Data/Processed/Botnet_Y.csv",index=False)