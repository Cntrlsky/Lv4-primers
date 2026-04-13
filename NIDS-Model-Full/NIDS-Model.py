#All  Imports
from importlib.resources import path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#Data is split for traning and testing purposes 
data01=pd.read_csv("NIDS-Model-Full/NSL_KDD_Test.csv",header=None)
data11=pd.read_csv("NIDS-Model-Full/NSL_KDD_Train.csv",header=None)

#This makess the 42 coloum of the data into the class as there is no header
data02=data01.rename(columns={41:'class'})
data12=data11.rename(columns={41:'class'})

#Data Duplication removal
data03=data02.drop_duplicates()
data13=data12.drop_duplicates()

#Missing Value removel (If too much data is removed use the other value fill meathod (ex:mean,median,mode))
data04=data03.dropna(subset=['class'])
data14=data13.dropna(subset=['class'])

#X is Input(Training) and  Questions(Test)
#Y is Output(Training) and Answers(Test)
X_train=data14.drop('class',axis=1)
Y_train=data14['class']
X_test=data04.drop('class',axis=1)
Y_test=data04['class']

#Data Encoding (Combine the training and testing data and making it into string)
X_combined=pd.concat([X_train, X_test], ignore_index=True)
X_combined=pd.get_dummies(X_combined, drop_first=True)
X_combined.columns=X_combined.columns.astype(str)

#Split to train and test
X_train=X_combined.iloc[:len(X_train), :]
X_test=X_combined.iloc[len(X_train):, :]

#this is the model it self
model=RandomForestClassifier()

#this is the code to train the model
model.fit(X_train,Y_train)

#this is the code to test the model
Predict=model.predict(X_test)
print("This models accuracy Score is",accuracy_score(Y_test,Predict))
print("This Models Precision Score is", precision_score(Y_test,Predict,average='weighted',zero_division=0))
print("This Models Recall Score is", recall_score(Y_test,Predict,average='weighted',zero_division=0))
print("This Models F1 Score is:", f1_score(Y_test,Predict,average='weighted',zero_division=0))

#change whats after the r to wherever you want to save the .pkl file
joblib.dump(model,r"C:\Users\hamza\Desktop\My Projects\Year Foundation\Lv4-Primers-Project\Project\PKL Files\model-one-NIS.pkl")