from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, roc_curve
from Train_Generator import Dataloader
import pandas as pd
from tqdm import tqdm

ID = "18-47-55"
folder = "2020-05-04_{}\\".format(ID)

# These are the CIFAR10 class labels from the training data (in order from 0 to 9)
class_labels = [
    "noise",
    "orca",
]
def append_to_list(path_wav,file_csv):
        list_f = []
        csv = pd.read_csv(file_csv)
        for file in csv.file_name.values:
            list_f.append(path_wav+file)
        return list_f

test_files = []

path ="C:\\myProjects\\THESIS\\csv\\ORCASPOT_csv\\"
#random_csv = path + "ROS_lab.csv"
#test_files += append_to_list(path,random_csv)
DeepAl_wav = "C:\\myProjects\\THESIS\\DeepAL_ComParE\\DeepAL_ComParE\\ComParE2019_OrcaActivity\\wav\\"
DeepAl_csv = "C:\\myProjects\\THESIS\\DeepAL_ComParE\\DeepAL_ComParE\\ComParE2019_OrcaActivity\\lab\\test_lab.csv"
test_files += append_to_list(DeepAl_wav,DeepAl_csv)

aud_wav = "C:\\myProjects\\THESIS\\Orchive\\audible\\"
aud_lab = aud_wav + "test_lab.csv"
test_files += append_to_list(aud_wav,aud_lab)

#AEOTD_wav = path + "AEOTD\\AEOTD\\"
#AEOTD_lab= path + "AEOTD\\AEOTD_label\\"
#test_files += append_to_list(AEOTD_wav,AEOTD_lab+"test_label.csv")

#DLFD = path + "DLFD\\"
#test_files += append_to_list(DLFD, DLFD + "test_label.csv")

#OAC_wav = path + "OAC\\OAC\\"
#OAC_lab = path + "OAC\\"
#test_files += append_to_list(OAC_wav, OAC_wav + "new_test_label.csv")

# Load the json file that contains the model's structure
f = Path(folder + "model_structure_{}.json".format(ID))
model_structure = f.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights(folder + "best_model_{}.h5".format(ID))

def update_file_to_label(file_to_label,wav_path,csv_path):
    csv = pd.read_csv(csv_path)
    file_to_label.update({wav_path + k: v for k, v in zip(csv.file_name.values,csv.label.values)})
    return file_to_label

file_to_label = {}
#update_file_to_label(file_to_label,path,random_csv)
#update_file_to_label(file_to_label,AEOTD_wav,AEOTD_lab+"test_label.csv")
#update_file_to_label(file_to_label,DLFD, DLFD + "test_label.csv")
#update_file_to_label(file_to_label,OAC_wav, OAC_wav + "new_test_label.csv")
update_file_to_label(file_to_label,DeepAl_wav,DeepAl_csv)
update_file_to_label(file_to_label,aud_wav,aud_lab)

label_to_int = {k: v for v, k in enumerate(class_labels)}
print('label_to_int ', label_to_int)

file_to_int = {k: label_to_int[v] for k, v in file_to_label.items()}

dl = Dataloader(file_to_int,False)

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

for i in file_to_label:
    file_name.append(i)
    labels.append(file_to_label[i])

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
