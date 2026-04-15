#All imported code needed for the code to run
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

#Data is split into attack and normal I will combine them later on
data01=pd.read_csv("Data/Raw/CTU13_Attack_Traffic.csv")
data11=pd.read_csv("Data/Raw/CTU13_Normal_Traffic.csv")

#Duplicate data removal
data02=data01.drop_duplicates()
data12=data11.drop_duplicates()

#Removing any values with a missing Label
data03=data02.dropna(subset=['Label'])
data13=data12.dropna(subset=['Label'])

#Combined the two csv files then encoded them , put label in Y and rest in X
X_combined=pd.concat([data03,data13])
X=pd.get_dummies(X_combined.drop(columns=['Label']))
Y=X_combined['Label']

#X and Y are split into traning and testing at a 80 20 split
X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y
    ,test_size=0.20
)

#This is the model itself
model=RandomForestClassifier(random_state=2007)

#This is the code to train the model
model.fit(X_train,Y_train)

#This is the code to test the model
Predict=model.predict(X_test)
print("This models accuracy Score is",accuracy_score(Y_test,Predict))
print("This Models Precision Score is", precision_score(Y_test,Predict,average='weighted'))
print("This Models Recall Score is", recall_score(Y_test,Predict,average='weighted'))
print("This Models F1 Score is:", f1_score(Y_test,Predict,average='weighted'))

#This is the code to save the .pkl file
joblib.dump(model,r"C:\Users\hamza\Desktop\My Projects\Year Foundation\Lv4-Primers-Project\Project\PKL Files\model-4-BotnetDetec.pkl")