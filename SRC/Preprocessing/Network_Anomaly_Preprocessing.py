import pandas as pd
from SRC.Feature_Engineering.Network_Feature_Engineering import Network_Feature_Engineering as NtWrk

data=pd.read_csv("Data/Raw/embedded_system_network_security_dataset.csv")

data01=data.drop_duplicates()

data02=data01.dropna(subset=['label'])

X=NtWrk(data02)
Y=data02['label']

X.to_csv("Data/Processed/Network_AnomalyX.csv",index=False)
Y.to_csv("Data/Processed/Network_AnomalyY.csv",index=False)