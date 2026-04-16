import pandas as pd

def Network_Feature_Engineering (data: pd.DataFrame):
    X=pd.get_dummies(data.drop(columns=["label"]))
    return X