from Train_Generator import *

from sklearn.cluster import KMeans, SpectralClustering
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import librosa.display
import os
import re

ap = argparse.ArgumentParser()

ap.add_argument("-cc","--cluster_csv", type=str,
                help="path for cluster csv")

ap.add_argument("--freq-compress", type=str, default='linear', help="define compression type")

ap.add_argument("--res-dir", type=str, help="path to model results plot,weights,checkpoints etc.")

ARGS = ap.parse_args()

df = pd.read_csv(ARGS.cluster_csv)

path = ARGS.res_dir + "SPECTRAL_CLUSTERS_nobot/"
if not os.path.isdir(path):
    os.mkdir(path)

dl = Dataloader(True, ARGS.freq_compress)

def getID(file):
    split = file.split("/")
    if re.findall(".aiff",file):
        ID = split[len(split) - 1].replace(".aiff", ".png")
    else:
        ID = split[len(split) - 1].replace(".wav", ".png")

    return ID

def plotspec(spec,ID,cluster):
    plt.subplot(111)
    librosa.display.specshow(spec.T, sr=44100, x_axis='time', y_axis=ARGS.freq_compress)
    plt.title(str(cluster))
    plt.savefig(ID)
    plt.close()

clusters = {}
for i,j in enumerate(df.values):
    cluster = j[2] #change to 3 for kmeans
    file = j[0]

    if cluster not in clusters.keys():
        clusters.update({cluster: [file]})
    else:
        clusters[cluster].append(file)


for c in clusters:
    cpath = path + str(c) + "/"
    if not os.path.isdir(cpath):
        os.mkdir(cpath)

    for f in clusters[c]:
        ID = getID(f)
        spec = dl.load_audio_file(f)
        plotspec(spec,cpath + ID,c)


