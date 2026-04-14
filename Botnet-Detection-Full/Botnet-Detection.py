#all imports needed for all five models
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

#change the data.csv to the name of your data file. Make sure it is in the same directory as this code.
data01=pd.read_csv("Botnet-Detection-Full/CTU13_Attack_Traffic.csv")
data11=pd.read_csv("Botnet-Detection/CTU13_Normal_Traffic.csv")

#Data Duplication 
data02=data01.drop_duplicates()
data12=data11.drop_duplicates()

#Missing Value removel (If too much data is removed use the other value fill meathod (ex:mean,median,mode))
data03=data02.dropna(subset=['Label'])
data13=data12.dropna(subset=['Label'])


X_combined=pd.concat([data03,data13])
X=pd.get_dummies(X_combined.drop(columns=['Label']))
Y=X_combined['Label']

#X is Input(Training) and  Questions(Test)
#Y is Output(Training) and Answers(Test)
#Make sure to change 'Cheating-data' to the name of the column that has the answers in it. This is the column we are trying to predict.
X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y
    ,test_size=0.20
    ,random_state=2007
)

#this is the model it self i dont belive we will need to change this as random forest is the best for anomaly detection which is the basis of all five of our models
model=RandomForestClassifier()

#this is the code to train the model
model.fit(X_train,Y_train)

#this is the code to test the model
Predict=model.predict(X_test)
print("This models accuracy Score is",accuracy_score(Y_test,Predict))
print("This Models Precision Score is", precision_score(Y_test,Predict,average='weighted'))
print("This Models Recall Score is", recall_score(Y_test,Predict,average='weighted'))
print("This Models F1 Score is:", f1_score(Y_test,Predict,average='weighted'))

#change the # to the number of the model
joblib.dump(model,r"C:\Users\hamza\Desktop\My Projects\Year Foundation\Lv4-Primers-Project\Project\PKL Files\model-four-BotnetDetec.pkl")