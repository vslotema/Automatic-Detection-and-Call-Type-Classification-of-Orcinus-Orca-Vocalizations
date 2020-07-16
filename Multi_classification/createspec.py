import pandas as pd
import argparse
from Train_Generator import *
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


ap = argparse.ArgumentParser()
ap.add_argument("--res-dir",type=str, help="directory of model results")
ap.add_argument("--freq-compress",type=str,default="linear", help="linear or mel compression")
ARGS = ap.parse_args()

results = pd.read_csv(ARGS.res_dir + "res_test.csv")

def createFileName(file,lab,pred_lab):
    split = file.split("/")
    new = split[len(split)-1].replace(".wav",".PNG")
    print("lab:{} pred_lab:{}".format(lab,pred_lab))
    if lab is pred_lab:
        new = ARGS.res_dir + "TRUE/" + new
    else:
        new = ARGS.res_dir + "FALSE/" + new
    return new

dl = Dataloader(True,ARGS.freq_compress)
for i,j in enumerate(results.values):
    file = j[0]
    spec = dl.load_audio_file(file)
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    librosa.display.specshow(spec, sr=44100, ax=ax, x_axis='time', y_axis=ARGS.freq_compress)
    save = createFileName(file,j[2],j[3])
    fig.savefig(save)
    plt.close()
