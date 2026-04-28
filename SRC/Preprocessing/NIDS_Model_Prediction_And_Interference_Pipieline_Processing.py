import pandas as pd
from sklearn.model_selection import train_test_split

def NIDS_Model_Prediction_And_Interference_Pipieline_Processing() :

    dataTest=pd.read_csv("Data/Processed/NIDS_Model_Test.csv")

    X=dataTest.drop(columns=["class"])
    Y=dataTest["class"]

    X_PrdIntr,X_useless,Y_PrdIntr,Y_useless=train_test_split(X,
                                                            Y,
                                                            test_size=0.6,
                                                            random_state=2007
                                                            )

    X_PrdIntr.to_csv("Data/Processed/NIDS_PrdIntr_X.csv",index=False)
    Y_PrdIntr.to_csv("Data/Processed/NIDS_PrdIntr_Y.csv",index=False)