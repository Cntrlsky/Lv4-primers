import pandas as pd
import joblib
from SRC.UTILS.Full import Full

X=pd.read_csv("Data/Processed/Botnet_X.csv")
Y=pd.read_csv("Data/Processed/Botnet_Y.csv")

model=Full(X,Y)

joblib.dump(model,"SRC/Models/Botnet_Model.pkl")