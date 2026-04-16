import pandas as pd

def Botnet_Feature_Engineering(data:pd.DataFrame):
    X=pd.get_dummies(data.drop(columns=["Label"]))
    return X