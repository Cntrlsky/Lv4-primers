import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
#1. Load Data and Model
X=pd.read_csv("Data/Processed/Botnet_X.csv")
y_true=pd.read_csv("Data/Processed/Botnet_Y.csv")
model=joblib.load("SRC/Models/Botnet_Model.pkl")
#2. Get Predictions
y_pred=model.predict(X)

#3. Generate Confusion Matrix 
cm=confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.title('Botnet Model Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig("Visualizations/Botnet/Botnet_Confusion_Matrix.png")
plt.show()
#4. Generate Classification Report 
print("Botnet Classification Report:\n")
print(classification_report(y_true,y_pred))

#Botnet ROC-AUC Curve
from sklearn.metrics import roc_curve, auc
import numpy as np
#1. Convert labels to binary (Using 'Label' column and numeric 1)
y_true_binary=np.where(y_true['Label']==1,1,0)
y_score=model.predict_proba(X)[:,1]
#2. Calculate ROC
fpr,tpr,thresholds=roc_curve(y_true_binary,y_score)
roc_auc=auc(fpr,tpr)

#3. Plot 
plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,color='darkorange',lw=2,label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
plt.title('Botnet Model ROC-AUC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig("Visualizations/Botnet/Botnet_ROC_Curve.png")
plt.show()

#Cross Validation Results
from sklearn.model_selection import cross_val_score
#1. Run 5-Fold Cross-Validation 
print("\nStarting 5-Fold Cross-Validation...")
cv_scores=cross_val_score(model,X,y_true.values.ravel(),cv=5)

#2. Output Results
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")

#3. Save results to a text file
with open("Visualizations/Botnet/Botnet_CV_Results.txt","w") as f:
    f.write(f"Cross-Validation Scores: {cv_scores}\n")
    f.write(f"Mean Accuracy: {cv_scores.mean()}\n")
    f.write(f"Std Dev: {cv_scores.std()}\n")

#Comparison Table with Baseline Models
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
#1. Create and train a "dumb" baseline model
print("Baseline Comparison")
dummy_clf=DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X,y_true.values.ravel()) 
dummy_preds=dummy_clf.predict(X)

#2. Calculate Accuracies
baseline_acc=accuracy_score(y_true,dummy_preds)
rf_acc=cv_scores.mean()

#3. Print Table
print(f"{'Model Type':<20} | {'Accuracy Score'}")
print("-" * 40)
print(f"{'Dummy Baseline':<20} | {baseline_acc:.4f}")
print(f"{'Random Forest (Yours)':<20} | {rf_acc:.4f}")
print("-" * 40)
print(f"Improvement: +{(rf_acc-baseline_acc)*100:.2f}%\n")

#4. Save proof
with open("Visualizations/Botnet/Botnet_Baseline_Comparison.txt","w") as f:
    f.write(f"Dummy Baseline Accuracy: {baseline_acc:.4f}\n")
    f.write(f"Random Forest Accuracy: {rf_acc:.4f}\n")