import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from SRC.UTILS.Full import Full

def test_full_function():
    X_train=pd.DataFrame({
        "Data_Limit":[3,5,6,2,3,7,12,22,40,23,31],
        "Data_Usage":[3,4,3,2,1,7,12,20,35,23,31]
    })
    
    X_test=pd.DataFrame({
        "Data_Limit":[12,55,6,7],
        "Data_Usage":[12,2,6,6]
    })

    Y_train=pd.Series([1,0,0,1,0,1,1,0,0,1,1])

    Y_test=pd.Series([1,0,1,0])

    model = Full(X_train,Y_train,X_test,Y_test)

    assert model is not None

    predictions=model.predict(X_test)

    assert len(predictions)==len(X_test)