import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score 
from sklearn.metrics import f1_score
import time

X=pd.read_csv("Data/Processed/Network_Anomaly_PrdIntr_X.csv")
Y=pd.read_csv("Data/Processed/Network_Anomaly_PrdIntr_Y.csv")

model=joblib.load("SRC/Models/Network_Anomaly_Model.pkl")

start_time=time.time()
Predict=model.predict(X)
end_time=time.time()
inference_time=end_time-start_time

print("This models accuracy Score is", accuracy_score(Y,Predict))
print("This Models Precision Score is", precision_score(Y,Predict,average='weighted',zero_division=0))
print("This Models Recall Score is", recall_score(Y,Predict,average='weighted',zero_division=0))
print("This Models F1 Score is:", f1_score(Y,Predict,average='weighted',zero_division=0))
print("Inference time is: ",inference_time," seconds")