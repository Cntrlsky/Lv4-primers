from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def Full(X_train,Y_train,X_test,Y_test):

    model=RandomForestClassifier(random_state=2007)

    if Y_train.ndim !=1:
        Y_train=Y_train.squeeze()
    
    if Y_test.ndim !=1:
        Y_test=Y_test.squeeze()

    model.fit(X_train,Y_train)

    Predict=model.predict(X_test)
    print("This models accuracy Score is", accuracy_score(Y_test,Predict))
    print("This Models Precision Score is", precision_score(Y_test,Predict,average='weighted',zero_division=0))
    print("This Models Recall Score is", recall_score(Y_test,Predict,average='weighted',zero_division=0))
    print("This Models F1 Score is:", f1_score(Y_test,Predict,average='weighted',zero_division=0))

    return model