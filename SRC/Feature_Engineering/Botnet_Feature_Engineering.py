import pandas as pd

# Here we remove the Label column and encode categorical values into binary columns
def Botnet_Feature_Engineering(data:pd.DataFrame):
    X=pd.get_dummies(data.drop(columns=["Label"]))
    return X