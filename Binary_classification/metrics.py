from OrganizeData import *
from keras.models import model_from_json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve,roc_auc_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--res-dir",type=str, help="directory of model results")
ap.add_argument("--data-dir",type=str, help="directory of the data's csv files")
ap.add_argument("--pos-label",type=str,default ="orca",help = "decide on the positive label")
ARGS = ap.parse_args()
path = ARGS.res_dir

results = pd.read_csv(path + 'res_test.csv')
f= open(path + "scores.txt","w+")
print(results)

class_labels = getUniqueLabels(ARGS.data_dir)
print(class_labels)

################## CONFUSION MATRIX
cm =confusion_matrix(results.label.values, results.pred_label.values,labels=class_labels)
print("cm ",cm)
cm_df = pd.DataFrame(cm,
                     index = class_labels,
                     columns = class_labels)

fig = plt.figure(figsize=(5,5))
sns.heatmap(cm_df, annot=True,fmt='d')
plt.title('Test Accuracy: {0:.3f}'.format(accuracy_score(results.label.values, results.pred_label.values)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
fig.savefig(path + 'confusion_matrix.png')

################## PRECISION
prec = precision_score(results.label.values,results.pred_label.values,pos_label=ARGS.pos_label)
print('Precision: {:.3f}'.format(prec*100))
f.write('Precision: {:.3f}\n'.format(prec*100))

################# RECALL
recall = recall_score(results.label.values,results.pred_label.values,pos_label=ARGS.pos_label)
print('Recall: {:.3f}'.format(recall*100))
f.write('Recall: {:.3f}\n'.format(recall*100))

################ F1 SCORE
f1 = f1_score(results.label.values,results.pred_label.values,pos_label=ARGS.pos_label)
print('F1 score: {:.3f}'.format(f1*100))
f.write('F1 score: {:.3f}\n'.format(f1*100) )

############## ROC CURVE
auc_score = 0
try:
    auc_score = roc_auc_score(results.label.values,results.pred_score.values)
except ValueError:
    pass
print('AUC: {:.3f}'.format(auc_score*100))
f.write('AUC: {:.3f}'.format(auc_score*100))
fpr, tpr, thresholds = roc_curve(results.label.values,results.pred_score.values,pos_label = ARGS.pos_label)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.5f)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()

#fig = plt.figure(figsize=(5,5))
#fpr, tpr, thresholds = roc_curve(results.label.values,results.pred_score.values,pos_label = ARGS.pos_label)
#sns.lineplot([0,1], [0,1], linestyle='--')

#sns.lineplot(fpr,tpr,marker='.')
#plt.legend('AUC {}'.format(auc_score),loc='best')
#plt.show()
plt.savefig(path + "roc_curve.png")






##############
