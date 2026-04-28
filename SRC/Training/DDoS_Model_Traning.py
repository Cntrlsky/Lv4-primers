import pandas as pd
import joblib
from SRC.UTILS.Full import Full

def DDoS_Model_Traning() :

    X=pd.read_csv("Data/Processed/DDoS_Model_Processed_X.csv")
    Y=pd.read_csv("Data/Processed/DDoS_Model_Processed_Y.csv")

    model=Full(X,Y)
           
    joblib.dump(model,"SRC/Models/DDoS_Model.pkl")