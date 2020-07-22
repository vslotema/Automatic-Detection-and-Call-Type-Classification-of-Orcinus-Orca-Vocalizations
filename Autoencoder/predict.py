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
import cv2
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir",type=str,help="path for training and val files")
ap.add_argument("--res-dir",type=str, help="directory of model results")
ap.add_argument("--freq-compress",type=str,default="linear", help="linear or mel compression")
ap.add_argument("--batch",type=int,default=16,help="determine size of batch")
ARGS = ap.parse_args()

folder = ARGS.res_dir

data_dir = ARGS.data_dir
test_files, file_to_label_test = findcsv("test",data_dir)
#class_labels = getUniqueLabels(data_dir)

# Load the json file that contains the model's structure
f = Path(folder + "model_structure.json")
model_structure = f.read_text()

# Recreate the Keras model object from the json data
autoencoder= model_from_json(model_structure)

# Re-load the model's trained weights
autoencoder.load_weights(folder + "best_model.h5")


# use the convolutional autoencoder to make predictions on the
# testing images, then initialize our list of output images
print("[INFO] making predictions...")
dl = Dataloader(False,freq_compress=ARGS.freq_compress)
_original = None
decoded = None

for batch_files in tqdm(dl.chunker(test_files, size=ARGS.batch), total=math.ceil(len(list(test_files)) // ARGS.batch)):
         batch_data = [dl.load_audio_file(fpath) for fpath in batch_files]
         batch_data = np.array(batch_data)[:, :, :, np.newaxis]
         print("batch data shape ", batch_data.shape)
         preds = autoencoder.predict(batch_data)
         if _original is None:
             _original = batch_data
         else:
             _original = np.concatenate((_original,batch_data))
             #_original.append(batch_data)
         print("orginal shape ", _original.shape)
         print("orginal type ", type(_original))
         if decoded is None:
             decoded = preds
         else:
             decoded = np.concatenate((decoded,preds))
             #decoded.append(preds)
         print("decoded shape ", decoded.shape)
         print("decoded type ", type(decoded))


outputs = None

def getID(test_file):
    split = test_file.split("/")
    ID = split[len(split)-1].replace(".wav","")
    return ID

def spec(spec1,spec2,ID):
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    librosa.display.specshow(spec1, sr=44100, ax=ax, x_axis='time', y_axis=ARGS.freq_compress)
    plt.title("encoded")

    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(112)
    librosa.display.specshow(spec2, sr=44100, ax=ax, x_axis='time', y_axis=ARGS.freq_compress)
    plt.title("decoded")
    fig.savefig(ID)
    plt.close()

# loop over our number of output samples
for i in range(0, len(test_files)-1):
    # grab the original image and reconstructed image
    #original = (_original[i] * 255).astype("uint8")
    #recon = (decoded[i] * 255).astype("uint8")
    ID = getID(test_files[i])
    # stack the original and reconstructed image side-by-side
    #output = np.hstack([original, recon])
    print("shape _original before ", _original[i].shape)
    original = np.squeeze(_original[i],axis=2)
    print("original shape ", original.shape)
    recon = np.squeeze(decoded[i],axis=2)
    print("dec shape ", recon.shape)
    spec(original,recon,folder + ID)
    #spec(recon,folder + ID + "_decoded.png")
   # cv2.imwrite(folder +"{}.png".format(ID),output)


