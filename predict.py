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

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir",type=str,help="path for training and val files")
ap.add_argument("--res-dir",type=str, help="directory of model results")
ap.add_argument("--freq-compress",type=str,default="linear", help="linear or mel compression")
ARGS = ap.parse_args()

folder = ARGS.res_dir

# These are the CIFAR10 class labels from the training data (in order from 0 to 9)
class_labels = [
    "noise",
    "orca",
]

data_dir = ARGS.data_dir
test_files, file_to_label_test = findcsv("test",data_dir)

# Load the json file that contains the model's structure
f = Path(folder + "model_structure.json")
model_structure = f.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights(folder + "best_model.h5")

label_to_int = {k: v for v, k in enumerate(class_labels)}
print('label_to_int ', label_to_int)

file_to_int = {k: label_to_int[v] for k, v in file_to_label_test.items()}

dl = Dataloader(file_to_int,False,freq_compress=ARGS.freq_compress)

bag = 3

array_preds = 0

for i in tqdm(range(bag)):

    list_preds = []

    for batch_files in tqdm(dl.chunker(test_files, size=16), total=len(list(test_files)) // 16):
        batch_data = [dl.load_audio_file(fpath) for fpath in batch_files]
        batch_data = np.array(batch_data)[:, :, :, np.newaxis]
        preds = model.predict(batch_data).tolist()
        #print('preds ', preds)
        list_preds += preds

    # In[21]:

    array_preds += np.array(list_preds) / bag

pred_labels = []
for i in array_preds:
    if i <= 0.5:
        pred_labels.append("noise")
    else:
        pred_labels.append("orca")

file_name = []
labels =[]

for i in file_to_label_test:
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
