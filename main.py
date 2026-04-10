#This is the most basic code for a ML model 
#DO NOT EDIT COPY ONLY

#all imports needed for all five models
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#change the data.csv to the name of your data file. Make sure it is in the same directory as this code.
data1=pd.read_csv('data.csv')

#Data Encoding
data2=pd.get_dummies(data1, drop_first=True)

#Data Duplication 
data3=data2.drop_duplicates()

#Missing Value removel (If too much data is removed use the other value fill meathod (ex:mean,median,mode))
data=data3.dropna(subset=['Cheating-data'])

#X is Input(Training) and  Questions(Test)
#Y is Output(Training) and Answers(Test)
#Make sure to change 'Cheating-data' to the name of the column that has the answers in it. This is the column we are trying to predict.
X = data.drop('Cheating-data', axis=1)
Y = data['Cheating-data']

#This will split the data 70 traning and 30 testing(random state can be any number just keep it consistent for all five models)
X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y 
    ,test_size=0.3
    ,random_state=2007)

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
joblib.dump(model, "model#.pkl")