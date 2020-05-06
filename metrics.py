from keras.models import model_from_json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve,roc_auc_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

ID = "18-47-55"
path = "2020-05-04_{}\\".format(ID)

results = pd.read_csv(path + 'res_test.csv')
f= open(path + "scores.txt","w+")
print(results)



################## CONFUSION MATRIX
cm =confusion_matrix(results.label.values, results.pred_label.values)
cm_df = pd.DataFrame(cm,
                     index = ['noise','orca'],
                     columns = ['noise','orca'])

fig = plt.figure(figsize=(5,5))
sns.heatmap(cm_df, annot=True,fmt='d')
plt.title('SVM Linear Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(results.label.values, results.pred_label.values)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
fig.savefig(path + 'confusion_matrix.png')

################## PRECISION
prec = precision_score(results.label.values,results.pred_label.values,pos_label='orca')
print('Precision: {:.3f}'.format(prec*100))
f.write('Precision: {:.3f}\n'.format(prec*100))

################# RECALL
recall = recall_score(results.label.values,results.pred_label.values,pos_label='orca')
print('Recall: {:.3f}'.format(recall*100))
f.write('Recall: {:.3f}\n'.format(recall*100))

################ F1 SCORE
f1 = f1_score(results.label.values,results.pred_label.values,pos_label='orca')
print('F1 score: {:.3f}'.format(f1*100))
f.write('F1 score: {:.3f}\n'.format(f1*100) )

############## ROC CURVE
fig = plt.figure(figsize=(5,5))
fpr, tpr, thresholds = roc_curve(results.label.values,results.pred_score.values,pos_label = 'orca')
sns.lineplot([0,1], [0,1], linestyle='--')
sns.lineplot(fpr,tpr,marker='.')
plt.show()
fig.savefig(path + "roc_curve.png")
auc_score = roc_auc_score(results.label.values,results.pred_score.values)
print('AUC: {:.3f}'.format(auc_score*100))
f.write('AUC: {:.3f}'.format(auc_score*100))

##############
