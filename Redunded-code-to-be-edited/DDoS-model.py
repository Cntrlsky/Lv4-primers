#All imports needed for the model
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

#Reading the csv
data01=pd.read_csv("Data/Raw/cicddos2019_dataset.csv")

#Duplicate data removal 
data02=data01.drop_duplicates()

#Removes any with missing value is class
data03=data02.dropna(subset=['Class'])

#encodes data in X and puts the header in Y
X=pd.get_dummies(data03.drop(columns=['Class','Label']))
Y=data03['Class']

#X and Y are split into traning and testing at a 80 20 split
X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y
    ,test_size=0.20
)

#This is the model itself
model=RandomForestClassifier(random_state=2007)

#This is the code to train the model
model.fit(X_train,Y_train)

#this is the code to test the model
Predict=model.predict(X_test)
print("This models accuracy Score is",accuracy_score(Y_test,Predict))
print("This Models Precision Score is", precision_score(Y_test,Predict,average='weighted'))
print("This Models Recall Score is", recall_score(Y_test,Predict,average='weighted'))
print("This Models F1 Score is:", f1_score(Y_test,Predict,average='weighted'))

#Code to save the model as a .pkl file
joblib.dump(model,r"C:\Users\hamza\Desktop\My Projects\Year Foundation\Lv4-Primers-Project\Project\PKL Files\model-2-DDoS.pkl")
