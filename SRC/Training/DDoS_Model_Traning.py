import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from SRC.UTILS.Full import Full

X=pd.read_csv("Data/Processed/DDos_Model_Processed_X.csv")
Y=pd.read_csv("Data/Processed/DDos_Model_Processed_Y.csv")

X_train,X_test,Y_train,Y_test = train_test_split(
    X,
    Y,
    test_size=0.2)


model=Full(X_train,Y_train,X_test,Y_test)

joblib.dump(model,r"C:\Users\hamza\Desktop\My Projects\Year Foundation\Lv4-Primers-Project\Project\PKL Files\DDoS_Model.pkl")
