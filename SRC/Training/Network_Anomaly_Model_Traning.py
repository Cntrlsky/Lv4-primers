import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from SRC.UTILS.Full import Full

X=pd.read_csv("Data/Processed/Network_AnomalyX.csv")
Y=pd.read_csv("Data/Processed/Network_AnomalyY.csv")

model=Full(X,Y)

joblib.dump(model,"SRC/Models/Network_Anomaly_Model.pkl")