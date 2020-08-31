import pandas as pd
import argparse
from Train_Generator import *
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import re


ap = argparse.ArgumentParser()
ap.add_argument("--csv",type=str, help="directory of csv")
ap.add_argument("--freq-compress",type=str,default="linear", help="linear or mel compression")
ARGS = ap.parse_args()

files = pd.read_csv(ARGS.csv)


def spec(spec1,spec2,spec3,spec4,ID):

    plt.subplot(111)
    librosa.display.specshow(spec1.T, sr=44100, x_axis='time', y_axis=ARGS.freq_compress)
    plt.title("original")
    plt.savefig(ID + "_01.PNG")
    plt.close()

    plt.subplot(111)
    librosa.display.specshow(spec2.T, sr=44100, x_axis='time', y_axis=ARGS.freq_compress)
    plt.title("added noise")
    plt.savefig(ID + "_02.PNG")
    plt.close()


    plt.subplot(111)
    librosa.display.specshow(spec3.T, sr=44100, x_axis='time', y_axis=ARGS.freq_compress)
    plt.title("pitch shift")
    plt.savefig(ID + "_03.PNG")
    plt.close()

    plt.subplot(111)
    librosa.display.specshow(spec4.T, sr=44100, x_axis='time', y_axis=ARGS.freq_compress)
    plt.title("time stretch")
    plt.savefig(ID + "_04.PNG")
    plt.close()

def getPathfile():
    split = ARGS.csv.split("/")
    path = ""
    for i in range(len(split) - 1):
        path += split[i] + "/"
    print(path)
    return path

def getID(test_file):
    name = test_file.replace(".wav","")
    ID = getPathfile() + name
    return ID

dl = Dataloader(True,ARGS.freq_compress)
for i,j in enumerate(files.values):
    ID = getID(j[0])
    spec1 = dl.load_audio_file(getPathfile() + j[0])
    print("spec1 ", spec1.shape)
    spec2 = dl.load_audio_file(getPathfile() + j[1])
    spec3 = dl.load_audio_file(getPathfile() + j[2])
    spec4 = dl.load_audio_file(getPathfile() + j[3])
    spec(spec1,spec2,spec3,spec4,ID)

