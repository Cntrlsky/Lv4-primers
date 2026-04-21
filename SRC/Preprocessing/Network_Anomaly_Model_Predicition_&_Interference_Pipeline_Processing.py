import pandas as pd
from sklearn.model_selection import train_test_split

dataX=pd.read_csv("Data/Processed/Network_AnomalyX.csv")
dataY=pd.read_csv("Data/Processed/Network_AnomalyY.csv")

X_train,X_test,Y_train,Y_test=train_test_split(dataX,
                                               dataY,
                                               test_size=0.2,
                                               random_state=2007
                                               )

X_PrdIntr,X_useless,Y_PrdIntr,Y_useless=train_test_split(X_test,
                                                        Y_test,
                                                        test_size=0.6,
                                                        random_state=2007
                                                        )

X_PrdIntr.to_csv("Data/Processed/Network_Anomaly_PrdIntr_X.csv",index=False)
Y_PrdIntr.to_csv("Data/Processed/Network_Anomaly_PrdIntr_Y.csv",index=False)