import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from SRC.Models.DDoS_Model_Model import model
from SRC.Models.DDoS_Model_Preprocessing import X_train, Y_train, X_test, Y_test

model.fit(X_train,Y_train)

Predict=model.predict(X_test)
print("This models accuracy Score is",accuracy_score(Y_test,Predict))
print("This Models Precision Score is", precision_score(Y_test,Predict,average='weighted'))
print("This Models Recall Score is", recall_score(Y_test,Predict,average='weighted'))
print("This Models F1 Score is:", f1_score(Y_test,Predict,average='weighted'))

joblib.dump(model,r"C:")
