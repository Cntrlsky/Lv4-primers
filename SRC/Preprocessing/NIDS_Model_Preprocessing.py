import pandas as pd
from SRC.Feature_Engineering.NIDS_Feature_Engineering import NIDS_Feature_Engineering

datatest=pd.read_csv("Data/Raw/NSL_KDD_Test.csv",header=None)
datatrain=pd.read_csv("Data/Raw/NSL_KDD_Train.csv",header=None)

datatest1=datatest.drop_duplicates()
datatrain1=datatrain.drop_duplicates()

datatest2=datatest1.rename(columns={41:'class'})
datatrain2=datatrain1.rename(columns={41:'class'})

datatest3=datatest2.dropna(subset=['class'])
datatrain3=datatrain2.dropna(subset=['class'])

datatrain5,Y_train,datatest5,Y_test=NIDS_Feature_Engineering(datatrain3,datatest3)

datatrain5['class']=Y_train.reset_index(drop=True)
datatest5['class']=Y_test.reset_index(drop=True)

datatest5.to_csv("Data/Processed/NIDS_Model_Test.csv",index=False)
datatrain5.to_csv("Data/Processed/NIDS_Model_Train.csv",index=False)