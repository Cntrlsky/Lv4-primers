import pandas as pd
import os

datatest=pd.read_csv("Data/Raw/NSL_KDD_Test.csv",header=None)
datatrain=pd.read_csv("Data/Raw/NSL_KDD_Train.csv",header=None)

datatest1=datatest.drop_duplicates()
datatrain1=datatrain.drop_duplicates()

datatest2=datatest1.rename(columns={41:'class'})
datatrain2=datatrain1.rename(columns={41:'class'})

datatest3=datatest2.dropna(subset=['class'])
datatrain3=datatrain2.dropna(subset=['class'])

X_train=datatrain3.drop(columns=['class'])
Y_train=datatrain3['class']

X_test=datatest3.drop(columns=['class'])
Y_test=datatest3['class']

data=pd.concat([X_train, X_test],ignore_index=True)
data=pd.get_dummies(data,drop_first=True)
data.columns=data.columns.astype(str)

datatrain5=data.iloc[:len(X_train),:].copy()
datatest5=data.iloc[len(X_train):,:].copy().reset_index(drop=True)

datatrain5['class']=Y_train.reset_index(drop=True)
datatest5['class']=Y_test.reset_index(drop=True)

datatest5.to_csv("Data/Processed/NIDS_Model_Test.csv",index=False)
datatrain5.to_csv("Data/Processed/NIDS_Model_Train.csv",index=False)