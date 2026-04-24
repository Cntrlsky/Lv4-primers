import pandas as pd

def NIDS_Feature_Engineering (train_data:pd.DataFrame,test_data:pd.DataFrame):
    #Here we put the data into x and y
    X_train=train_data.drop(columns=["class"])
    Y_train=train_data["class"]

    X_test=test_data.drop(columns=["class"])
    Y_test=test_data["class"]

    #Here we merge the two datasets and we encode them and turn all coloum names in X into string
    X=pd.concat([X_train, X_test], ignore_index=True)
    X=pd.get_dummies(X,drop_first=True)
    X.columns=X.columns.astype(str)

    #this splits X back into traning and testing by using the length of X_train and putting th erest into X_test
    datatrain5=X.iloc[:len(X_train),:].copy()
    datatest5=X.iloc[len(X_train):,:].copy().reset_index(drop=True)

    return datatrain5,Y_train.reset_index(drop=True),datatest5,Y_test.reset_index(drop=True)