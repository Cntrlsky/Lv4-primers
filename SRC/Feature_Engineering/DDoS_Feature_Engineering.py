import pandas as pd

def DDoS_Feature_Engineering(data: pd.DataFrame):
    X = pd.get_dummies(data.drop(columns=["Class","Label"]))
    return X