from keras.models import model_from_json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve,roc_auc_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--res-dir",type=str, help="directory of model results")
ARGS = ap.parse_args()
path = ARGS.res_dir

results = pd.read_csv(path + 'res_test.csv')
labels = pd.read_csv("/mnt/c/myProjects/THESIS/Data/PODS/train.csv")
f= open(path + "scores.txt","w+")
print(results)

def unique(list):
    unique = []
    for i in list:
        if i not in unique:
            unique.append(i)
    return unique

class_labels = unique(list(results.label.values))
print(class_labels)

################## CONFUSION MATRIX
cm =confusion_matrix(results.label.values, results.pred_1.values,labels=class_labels)
print("cm ",cm)
cm_df = pd.DataFrame(cm,
                     index = class_labels,
                     columns = class_labels)
fig = plt.figure(figsize=(5,5))
sns.heatmap(cm_df, annot=True,fmt='d')
plt.title('SVM Linear Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(results.label.values, results.pred_1.values)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
fig.savefig(path + 'confusion_matrix.png')

################# PRECISION,RECALL,F1
print(metrics.classification_report(results.label.values, results.pred_1.values, digits=3))
#################### AUC
auc_score = 0
try:
    auc_score = roc_auc_score(results.label.values,results.pred_1_score.values,max_fpr=1.0,multi_class='ovr')
except ValueError:
    pass

############## ROC CURVE
fig = plt.figure(figsize=(5,5))
fpr, tpr, thresholds = roc_curve(results.label.values,results.pred_1_score.values,pos_label=1)
sns.lineplot([0,1], [0,1], linestyle='--')
sns.lineplot(fpr,tpr,marker='.')
plt.show()
fig.savefig(path + "roc_curve.png")


##############
