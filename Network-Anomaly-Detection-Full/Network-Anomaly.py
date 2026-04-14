#All imports needed for this code
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

#Code to read the .csv
data01=pd.read_csv("Network-Anomaly-Detection-Full/embedded_system_network_security_dataset.csv")

#Duplicate data removal
data02=data01.drop_duplicates()

#Rows with missing labels are removed
data03=data02.dropna(subset=['label'])

#Data encoded and split into label and the rest of the data
X=pd.get_dummies(data03.drop(columns=['label']))
Y=data03['label']

#X and Y are split into training and testing at a 80-20 split
X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y
    ,test_size=0.20
)

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

#This is the code to save the model as a .pkl file
joblib.dump(model,r"C:\Users\hamza\Desktop\My Projects\Year Foundation\Lv4-Primers-Project\Project\PKL Files\model-3-NtwrkAnmly.pkl")
