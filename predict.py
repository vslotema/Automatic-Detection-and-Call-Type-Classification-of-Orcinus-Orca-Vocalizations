from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, roc_curve
from Train_Generator import Dataloader
import pandas as pd
from tqdm import tqdm

# These are the CIFAR10 class labels from the training data (in order from 0 to 9)
class_labels = [
    "noise",
    "orca",
]

split = "val"
AEOTD = "AEOTD\\"
path = "C:\\myProjects\\THESIS\\csv\\ORCASPOT_csv\\" + AEOTD
path_wav = path + AEOTD + AEOTD

folder = "2020-03-26_16-23-29\\"
# Load the json file that contains the model's structure
f = Path(folder + "model_structure_16-23-29.json")
model_structure = f.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights(folder + "weights_16-23-29.h5")

test_csv = pd.read_csv(path + "new_{}_label.csv".format(split))
#wav_path = "C:\\myProjects\\THESIS\\DeepAL_ComParE\\DeepAL_ComParE\\ComParE2019_OrcaActivity\\wav\\"

file_to_label = {path_wav + k: v for k, v in zip(test_csv.file_name.values, test_csv.label.values)}

# In[8]:
list_labels = sorted(list(set(test_csv.label.values)))  # Unique labels orca and noise
print('list_labels ', list_labels)


label_to_int = {k: v for v, k in enumerate(list_labels)}
print('label_to_int ', label_to_int)


file_to_int = {k: label_to_int[v] for k, v in file_to_label.items()}

dl = Dataloader(file_to_int,False)


bag = 3

array_preds = 0

for i in tqdm(range(bag)):

    list_preds = []

    for batch_files in tqdm(dl.chunker(path_wav + test_csv.file_name.values, size=16), total=len(list(test_csv)) // 16):
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

df = pd.DataFrame(test_csv.file_name.values, columns=["file_name"])
df['label'] = test_csv.label.values
df['pred_label'] = pred_labels
df['pred_score'] = array_preds


df.to_csv(path + "res_{}.csv".format(split), index=False)

results = pd.read_csv(path + "res_{}.csv".format(split))
print(confusion_matrix(results.label.values, results.pred_label.values))
print(matthews_corrcoef(results.label.values, results.pred_label.values))
