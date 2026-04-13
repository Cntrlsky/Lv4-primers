#This is the most basic code for a ML model 
#DO NOT EDIT COPY ONLY

#all imports needed for all five models
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#change the data.csv to the name of your data file. Make sure it is in the same directory as this code.
data01=pd.read_csv("Data")
data11=pd.read_csv("Data2",)

#Data Duplication 
data02=data01.drop_duplicates()
data12=data11.drop_duplicates()

#Missing Value removel (If too much data is removed use the other value fill meathod (ex:mean,median,mode))
data03=data02.dropna(subset=['class'])
data13=data12.dropna(subset=['class'])

#X is Input(Training) and  Questions(Test)
#Y is Output(Training) and Answers(Test)
#Make sure to change 'Cheating-data' to the name of the column that has the answers in it. This is the column we are trying to predict.
X_train=data13.drop('class',axis=1)
Y_train=data13['class']
X_test=data03.drop('class',axis=1)
Y_test=data03['class']

#Data Encoding (Combine the training and testing data and making it into string)
X_combined=pd.concat([X_train, X_test], ignore_index=True)
X_combined=pd.get_dummies(X_combined, drop_first=True)
X_combined.columns=X_combined.columns.astype(str)

#Split to train and test
X_train=X_combined.iloc[:len(X_train), :]
X_test=X_combined.iloc[len(X_train):, :]


#this is the model it self i dont belive we will need to change this as random forest is the best for anomaly detection which is the basis of all five of our models
model=RandomForestClassifier()

#this is the code to train the model
model.fit(X_train,Y_train)

#this is the code to test the model
Predict=model.predict(X_test)
print("This models accuracy Score is",accuracy_score(Y_test,Predict))
print("This Models Precision Score is", precision_score(Y_test,Predict,average='binary'))
print("This Models Recall Score is", recall_score(Y_test,Predict,average='binary'))
print("This Models F1 Score is:", f1_score(Y_test,Predict,average='binary'))

#change the # to the number of the model
joblib.dump(model,"model#.pkl")
