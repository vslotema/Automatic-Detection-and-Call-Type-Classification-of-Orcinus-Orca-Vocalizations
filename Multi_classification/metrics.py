from OrganizeData import *
from keras.models import model_from_json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve,roc_auc_score, accuracy_score,auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from scipy import interp
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--res-dir",type=str, help="directory of model results")
ap.add_argument("--data-dir",type=str, help="directory of data csv files")
ARGS = ap.parse_args()
path = ARGS.res_dir

results = pd.read_csv(path + 'res_test.csv')
f= open(path + "scores.txt","w+")

class_labels = getUniqueLabels(ARGS.data_dir)
int_to_label = {k:v for (k,v) in enumerate(class_labels)}
print("int to label ", int_to_label)
print(class_labels)
y_test = label_binarize(results.label.values,classes=class_labels)
print("Y ", y_test)
print("Y ", y_test.shape)
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
plt.close()

################# PRECISION,RECALL,F1
print(metrics.classification_report(results.label.values, results.pred_1.values,labels=class_labels, digits=3))

############## ROC Curve

scores = pd.read_csv(path + "scores.csv")
y_score = []
for i in scores.values:
    y_score.append(i.tolist())
y_score=np.array(y_score)
print("y score shape ", y_score.shape)

# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(class_labels)):
    #print("i:{} yt:{} ys:{}".format(i,y_test[:,i], y_score[:,i]))
    fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_score[:,i])
    try:
        roc_auc[i] = auc(fpr[i], tpr[i])
    except ValueError:
        pass
    #print("{} fpr:{} tpr:{} roc_auc:{}".format(int_to_label.get(i),fpr[i],tpr[i],roc_auc[i]))
print("got here")
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(np.concatenate(y_test).ravel(), np.concatenate(y_score).ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_labels))]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(class_labels)):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= (len(class_labels))

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue','yellow','red','blue',"violet","teal","greenyellow","coral","palegreen","slateblue"])
for i, color in zip(range(len(class_labels)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             .format(int_to_label.get(i), roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC multi-class')
plt.legend(loc="best")
plt.savefig(path + "roc_curve.png")
plt.show()
plt.close()

plt.figure(2)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC multi-class')
plt.legend(loc="best")
plt.savefig(path + "macro_micro_roc_curve.png")
plt.show()
plt.close()


macro_roc_auc_ovo = roc_auc_score(y_test, y_score, multi_class="ovo",
                                      average="macro")

weighted_roc_auc_ovo = roc_auc_score(y_test, y_score, multi_class="ovo",
                                     average="weighted")
macro_roc_auc_ovr = roc_auc_score(y_test, y_score, multi_class="ovr",
                                  average="macro")
weighted_roc_auc_ovr = roc_auc_score(y_test, y_score, multi_class="ovr",
                                     average="weighted")
print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
      "(weighted by prevalence)"
      .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
      "(weighted by prevalence)"
      .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))
