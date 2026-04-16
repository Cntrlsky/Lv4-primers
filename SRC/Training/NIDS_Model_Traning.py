import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

X=pd.read_csv("Data/Processed/NIDS_Model_Train.csv")
Y=pd.read_csv("Data/Processed/NIDS_Model_Test.csv")

X_train=X.drop('class',axis=1)
Y_train=X['class']
X_test=Y.drop('class',axis=1)
Y_test=Y['class']

model=RandomForestClassifier(random_state=2007)

model.fit(X_train,Y_train)

Predict=model.predict(X_test)
print("This models accuracy Score is", accuracy_score(Y_test,Predict))
print("This Models Precision Score is", precision_score(Y_test,Predict,average='weighted',zero_division=0))
print("This Models Recall Score is", recall_score(Y_test,Predict,average='weighted',zero_division=0))
print("This Models F1 Score is:", f1_score(Y_test,Predict,average='weighted',zero_division=0))

joblib.dump(model,r"C:\Users\hamza\Desktop\My Projects\Year Foundation\Lv4-Primers-Project\Project\PKL Files\NIDS_Model.pkl")