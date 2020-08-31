import pandas as pd
import argparse
from Train_Generator import *
import matplotlib.pyplot as plt
import librosa.display
from shutil import copyfile

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import re


ap = argparse.ArgumentParser()
ap.add_argument("--csv",type=str, help="directory of csv")
ap.add_argument("--freq-compress",type=str,default="linear", help="linear or mel compression")
ap.add_argument("--data-dir",type=str,default=None, help="path to data")
ap.add_argument("--res-dir",type=str,default=None,help="path to save the spectrograms")
ARGS = ap.parse_args()

files = pd.read_csv(ARGS.csv)


def plotspec(spec,ID):
    plt.subplot(111)
    librosa.display.specshow(spec.T, sr=44100, x_axis='time', y_axis=ARGS.freq_compress)
    plt.title("original")
    plt.savefig(ARGS.res_dir + ID)
    plt.close()


def getID(test_file,x):
    split = test_file.split("/")
    name = split[len(split)-1].replace(".wav","")
    ID = x + "_" + name
    return ID

def splitname(file):
    split = file.split("/")
    name = split[len(split) - 1]
    return name


dl = Dataloader(True,ARGS.freq_compress)
for i,j in enumerate(files.values):
    lab1 = j[1]
    lab2 = j[2]
    score = j[3]
    if lab1 != lab2:
        if lab1 == 'orca' and score < 0.10:
            ID = getID(j[0],"FN")
            spec = dl.load_audio_file(ARGS.data_dir + splitname(j[0]))
            plotspec(spec, ID)
        elif lab1 == 'noise' and score > 0.90:
            print(score)
            ID = getID(j[0],"FP")
            spec = dl.load_audio_file(ARGS.data_dir + splitname(j[0]))
            plotspec(spec, ID)





