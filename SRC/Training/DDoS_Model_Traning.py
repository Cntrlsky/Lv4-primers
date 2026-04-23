import pandas as pd
import joblib
from SRC.UTILS.Full import Full

X=pd.read_csv("Data/Processed/DDoS_Model_Processed_X.csv")
Y=pd.read_csv("Data/Processed/DDoS_Model_Processed_Y.csv")

model=Full(X,Y)

joblib.dump(model,r"C:\Users\hamza\Desktop\My Projects\Year Foundation\Lv4-Primers-Project\Project\SRC\Models\DDoS_Model.pkl")