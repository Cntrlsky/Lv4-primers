#This is the most basic code for a ML model 
#DO NOT EDIT COPY ONLY

#all imports needed for all five models
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#change the data.csv to the name of your data file. Make sure it is in the same directory as this code.
data = pd.read_csv('data.csv')

#X is Input(Training) and  Questions(Test)
#Y is Output(Training) and Answers(Test)
#Make sure to change 'Cheating-data1' to the name of the column that has the answers in it. This is the column we are trying to predict.
#Make sure to change 'Cheating-data2' to the name of the column that has the questions in it. This is the column we are using to predict the answers.
X = data.drop('Cheating-data1', axis=1)
Y = data['Cheating-data2']

#This will split the data 70 traning and 30 testing(random state can be any number just keep it consistent for all five models)
X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y 
    ,test_size=0.3
    ,random_state=2007)

#this is the model it self i dont belive we will need to change this
model=RandomForestClassifier()

#this is the code to train the model
model.fit(X_train,Y_train)

#this is the code to test the model
Predict=model.predict(X_test)
print("This models accuracy is",accuracy_score(Y_test,Predict))