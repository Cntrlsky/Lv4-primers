import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from SRC.UTILS.Full import Full

X=pd.read_csv("Data/Processed/NIDS_Model_Train.csv")
Y=pd.read_csv("Data/Processed/NIDS_Model_Test.csv")

X_train=X.drop('class',axis=1)
Y_train=X['class']
X_test=Y.drop('class',axis=1)
Y_test=Y['class']

model=Full(X_train,Y_train,X_test,Y_test)

joblib.dump(model,r"C:\Users\hamza\Desktop\My Projects\Year Foundation\Lv4-Primers-Project\Project\PKL Files\NIDS_Model.pkl")