#All  Imports needed for the model to work
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
data01=pd.read_csv("Zero-Day-Atk-Det/zero_day_attack_detection_dataset_V1-400k.csv")
data11=pd.read_csv("Zero-Day-Atk-Det/zero_day_attack_detection_dataset_V2_97k.csv")

#Data Duplication removal
data02=data01.drop_duplicates()
data12=data11.drop_duplicates()

#Rows without class are removed
data03=data02.dropna(subset=['Prediction'])
data13=data12.dropna(subset=['Prediction'])

#Drop to decrease leaks
data04=data03.drop(['Threat Level'
                    ,'Family'
                    ,'Application Layer Data'
                    ,'Event Description'
                    ,'User-Agent'
                    ,'Anomaly Score'
                    ,'Time'
                    ,'SeddAddress'
                    ,'ExpAddress'
                    ,'IP Address'
                    ,'Logistics ID'
                    ,'Session ID'
                    ], axis=1)
data14=data13.drop(['Threat Level'
                    ,'Family'
                    ,'Application Layer Data'
                    ,'Event Description'
                    ,'User-Agent'
                    ,'Anomaly Score'
                    ,'Time'
                    ,'SeddAddress'
                    ,'ExpAddress'
                    ,'IP Address'
                    ,'Logistics ID'
                    ,'Session ID'
                    ], axis=1)


X_train0=data04.drop('Prediction',axis=1)
Y_train=data04['Prediction']
X_test0=data14.drop('Prediction',axis=1)
Y_test=data14['Prediction']

combined_X=pd.concat([X_train0,X_test0],axis=0)
combined_X=pd.get_dummies(combined_X)

X_train=combined_X.iloc[:len(X_train0),:]
X_test=combined_X.iloc[len(X_test0):,:]

#This is the model it self
model=RandomForestClassifier(random_state=2007)

#This is the code to train the model
model.fit(X_train,Y_train)

#This is the code to test the model
Predict=model.predict(X_test)
print("This models accuracy Score is",accuracy_score(Y_test,Predict))
print("This Models Precision Score is", precision_score(Y_test,Predict,average='weighted',zero_division=0))
print("This Models Recall Score is", recall_score(Y_test,Predict,average='weighted',zero_division=0))
print("This Models F1 Score is:", f1_score(Y_test,Predict,average='weighted',zero_division=0))

#Change whats after the r to wherever you want to save the .pkl file
joblib.dump(model,r"C:\Users\hamza\Desktop\My Projects\Year Foundation\Lv4-Primers-Project\Project\PKL Files\model-5-Zero-Day-Atk-Det.pkl")