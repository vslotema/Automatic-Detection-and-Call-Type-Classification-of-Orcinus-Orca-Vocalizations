from keras.models import model_from_json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve,roc_auc_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

path = "2020-03-26_16-23-29\\"
results = pd.read_csv(path + 'res.csv')
print(results)



################## CONFUSION MATRIX
cm =confusion_matrix(results.label.values, results.pred_label.values)
cm_df = pd.DataFrame(cm,
                     index = ['noise','orca'],
                     columns = ['noise','orca'])

plt.figure(figsize=(5,5))
sns.heatmap(cm_df, annot=True,fmt='d')
plt.title('SVM Linear Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(results.label.values, results.pred_label.values)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

################## PRECISION
prec = precision_score(results.label.values,results.pred_label.values,pos_label='orca')
print('Precision: {:.3f}'.format(prec*100))

################# RECALL
recall = recall_score(results.label.values,results.pred_label.values,pos_label='orca')
print('Recall: {:.3f}'.format(recall*100))

################ F1 SCORE
f1 = f1_score(results.label.values,results.pred_label.values,pos_label='orca')
print('F1 score: {:.3f}'.format(f1*100))

############## ROC CURVE
fpr, tpr, thresholds = roc_curve(results.label.values,results.pred_score.values,pos_label = 'orca')
sns.lineplot([0,1], [0,1], linestyle='--')
sns.lineplot(fpr,tpr,marker='.')
plt.show()
auc_score = roc_auc_score(results.label.values,results.pred_score.values)
print('AUC: %.3f' % auc_score)

##############
