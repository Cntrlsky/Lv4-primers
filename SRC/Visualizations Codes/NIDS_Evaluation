import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
#_1. Load Data and Model
test_data=pd.read_csv("Data/Processed/NIDS_Model_Test.csv")
model=joblib.load("SRC/Models/NIDS_Model.pkl")
#_Drop the label column from X so the features match the model training
X=test_data.drop(columns=['class']) 
y_true=test_data['class']
#_2. Get Predictions
y_pred=model.predict(X)
# ___
#_3. Generate Confusion Matrix 
cm=confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm,annot=False,cmap='Blues')
plt.title('NIDS Multiclass Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig("Visualizations/NIDS/NIDS_Confusion_Matrix_Processed.png")
plt.show()
#_4. Generate Classification Report 
print("NIDS Classification Report:\n")
print(classification_report(y_true,y_pred))
# ___
#NIDS ROC-AUC Curve
from sklearn.metrics import roc_curve, auc
import numpy as np
#_1. Convert labels to binary: 0 for 'normal', 1 for anything else (Attack)
y_true_binary=np.where(y_true=='normal',0,1)
#_2. Calculate total attack probability
y_score=1-model.predict_proba(X)[:,0]
#_3. Calculate ROC
fpr,tpr,thresholds=roc_curve(y_true_binary,y_score)
roc_auc=auc(fpr,tpr)
#_4. Plot 
plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,color='darkorange',lw=2,label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
plt.title('NIDS Binary (Attack vs Normal) ROC-AUC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig("Visualizations/NIDS/NIDS_ROC_Curve_Binary.png")
plt.show()
# ___
#Cross Validation Results
from sklearn.model_selection import cross_val_score
#_1. Run 5-Fold Cross-Validation 
print("\nStarting 5-Fold Cross-Validation...")
cv_scores=cross_val_score(model,X,y_true.values.ravel(),cv=5)
#_2. Output Results
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
#_3. Save results
with open("Visualizations/NIDS/NIDS_CV_Results_Processed.txt","w") as f:
    f.write(f"Mean Accuracy: {cv_scores.mean()}\n")
    f.write(f"Std Dev: {cv_scores.std()}\n")