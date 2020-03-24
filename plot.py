from keras.models import model_from_json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, roc_curve,accuracy_score, plot_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

path = "2020-03-23_06-31-34\\"
results = pd.read_csv(path + 'res.csv')
print(results)
cm =confusion_matrix(results.label.values, results.pred_label.values)


################## CONFUSION MATRIX
cm_df = pd.DataFrame(cm,
                     index = ['noise','orca'],
                     columns = ['noise','orca'])

plt.figure(figsize=(5,5))
sns.heatmap(cm_df, annot=True,fmt='d')
plt.title('SVM Linear Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(results.label.values, results.pred_label.values)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
