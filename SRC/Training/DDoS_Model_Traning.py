import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X=pd.read_csv("Data/Processed/DDos_Model_Processed_X.csv")
Y=pd.read_csv("Data/Processed/DDos_Model_Processed_Y.csv").squeeze()

X_train,X_test,Y_train,Y_test = train_test_split(
    X,
    Y,
    test_size=0.2)

model=RandomForestClassifier(random_state=2007)

model.fit(X_train,Y_train)

Predict=model.predict(X_test)
print("This models accuracy Score is",accuracy_score(Y_test,Predict))
print("This Models Precision Score is", precision_score(Y_test,Predict,average='weighted'))
print("This Models Recall Score is", recall_score(Y_test,Predict,average='weighted'))
print("This Models F1 Score is:", f1_score(Y_test,Predict,average='weighted'))

joblib.dump(model,r"C:\Users\hamza\Desktop\My Projects\Year Foundation\Lv4-Primers-Project\Project\PKL Files\DDoS_Model.pkl")
