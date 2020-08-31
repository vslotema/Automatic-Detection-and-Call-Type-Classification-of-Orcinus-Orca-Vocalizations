import pandas as pd
import argparse
from Train_Generator import *
from OrganizeData import *
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import re


ap = argparse.ArgumentParser()
ap.add_argument("--data-dir",type=str, help="directory of data")
ap.add_argument("--freq-compress",type=str,default="linear", help="linear or mel compression")
ARGS = ap.parse_args()

train_f,train_ftl = findcsv("train", ARGS.data_dir)
val_f,val_ftl = findcsv("val", ARGS.data_dir)
test_f,test_ftl = findcsv("test", ARGS.data_dir)

calls = ["N01","N02","N03","N04","N05","N07","N08",
         "N09","N10","N11","N12","N13","N16","N17",
         "N18","N20","N23","N24","N25","N26","N28",
         "N29","N30","N32","N33","N34","N38","N41",
         "N44","N45","N47"]


def createFileName(file,lab):
    split = file.split("/")
    new = split[len(split)-1].replace(".wav",".PNG")
    call = False
    for i in calls:
        if re.findall(i,new):
            new = ARGS.data_dir + i + "/{}_{}".format(lab,new)
            call = True
            break
    if not call:
        new = ARGS.data_dir + "unknown_call" + "/{}_{}".format(lab,new)
    return new

dl = Dataloader(False,ARGS.freq_compress)
for f in train_f:
    spec = dl.load_audio_file(f).T
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    librosa.display.specshow(spec, sr=44100, ax=ax, x_axis='time', y_axis=ARGS.freq_compress)
    save = createFileName(f,train_ftl[f])
    fig.savefig(save)
    plt.close()

for f in test_f:
    spec = dl.load_audio_file(f).T
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    librosa.display.specshow(spec, sr=44100, ax=ax, x_axis='time', y_axis=ARGS.freq_compress)
    save = createFileName(f,test_ftl[f])
    fig.savefig(save)
    plt.close()

for f in val_f:
    spec = dl.load_audio_file(f).T
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    librosa.display.specshow(spec, sr=44100, ax=ax, x_axis='time', y_axis=ARGS.freq_compress)
    save = createFileName(f,val_ftl[f])
    fig.savefig(save)
    plt.close()
