from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, roc_curve
from Train_Generator import Dataloader
import pandas as pd
from tqdm import tqdm
from OrganizeData import *
import argparse
import math

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir",type=str,help="path for training and val files")
ap.add_argument("--res-dir",type=str, help="directory of model results")
ap.add_argument("--freq-compress",type=str,default="linear", help="linear or mel compression")
ap.add_argument("--batch",type=int,default=16,help="determine size of batch")
ARGS = ap.parse_args()

folder = ARGS.res_dir

data_dir = ARGS.data_dir
test_files, file_to_label_test = findcsv("test",data_dir)

def unique(list):
    unique = []
    for i in list:
        if i not in unique:
            unique.append(i)
        elif len(unique) == 2:
            break
    return sorted(unique)

class_labels = unique(list(file_to_label_test.values()))

# Load the json file that contains the model's structure
f = Path(folder + "model_structure.json")
model_structure = f.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights(folder + "best_model.h5")

label_to_int = {k: v for v, k in enumerate(class_labels)}
print('label_to_int ', label_to_int)
int_to_label = {label_to_int[class_labels[0]]:class_labels[0],label_to_int[class_labels[1]]:class_labels[1]}
print("int_to_label[0] ",int_to_label.get(0))
print("int_to_label[1] ",int_to_label.get(1))

file_to_int = {k: label_to_int[v] for k, v in file_to_label_test.items()}

dl = Dataloader(file_to_int,False,freq_compress=ARGS.freq_compress)

bag = 1

array_preds = 0

for i in tqdm(range(bag)):

    list_preds = []

    for batch_files in tqdm(dl.chunker(test_files, size=ARGS.batch), total=math.ceil(len(list(test_files)) // ARGS.batch)):
        batch_data = [dl.load_audio_file(fpath) for fpath in batch_files]
        batch_data = np.array(batch_data)[:, :, :, np.newaxis]
        preds = model.predict(batch_data).tolist()
        #print('preds ', preds)
        list_preds += preds

    # In[21]:

    array_preds += np.array(list_preds) / bag
print("array preds ", array_preds)


pred_labels = []
for i in array_preds:
    if i <= 0.5:
        pred_labels.append(int_to_label.get(0))
    else:
        pred_labels.append(int_to_label.get(1))

file_name = []
labels =[]

for i in test_files:
    file_name.append(i)
    labels.append(file_to_label_test[i])

df = pd.DataFrame(file_name, columns=["file_name"])
df['label'] = labels
df['pred_label'] = pred_labels
df['pred_score'] = array_preds


df.to_csv(folder + "res_test.csv", index=False)

results = pd.read_csv(folder + "res_test.csv")
printres="results file: {}".format(results.file_name.values)  + " label: {}".format(results.label.values)
print(printres)
print(confusion_matrix(results.label.values, results.pred_label.values))
print(matthews_corrcoef(results.label.values, results.pred_label.values))
