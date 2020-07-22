# In[0]:
from Train_Generator import *
from OrganizeData import *

from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import math
import random
from sklearn.manifold import TSNE
from tqdm import tqdm
from itertools import cycle
import numpy as np

from keras.models import model_from_json
from keras.layers import (
    Input,
    Dense,
    Flatten,
    AveragePooling2D
)
from keras import Model
import keras.backend as K

ap = argparse.ArgumentParser()

ap.add_argument("--n_threads", type=int, default=4, help="number of working threads")

ap.add_argument("--data-dir", type=str,
                help="path for training and val files")

ap.add_argument("--freq-compress", type=str, default='linear', help="define compression type")

ap.add_argument("-m", "--model", type=str, default=None,
                help="path to model checkpoint to load in case you want to resume a model training from where you left of")

ap.add_argument("--res-dir", type=str, help="path to model results plot,weights,checkpoints etc.")

ap.add_argument("-ie", "--initial-epoch", type=int, default=0,
                help="epoch to restart training at")

ap.add_argument("--batch", type=int, default=32, help="choose batch size")

ap.add_argument("--n-epochs", type=int, default=100, help="number of epochs")

ap.add_argument(
    "--lr",
    "--learning_rate",
    type=float,
    default=1e-5,
    help="Initial learning rate. Will get multiplied by the batch size.",
)

ARGS = ap.parse_args()

def setseed():
    seed = 10
    random.seed(seed)
    np.random.seed(seed)

def getFeatures(files,model):

    dl = Dataloader(file_to_label, False, freq_compress=ARGS.freq_compress)
    features = None
    labels = []
    for batch_files in tqdm(dl.chunker(files, size=ARGS.batch), total=math.ceil(len(list(files)) // ARGS.batch),
                            desc="running the model inference"):
        batch_data = [dl.load_audio_file(fpath) for fpath in batch_files]
        batch_data = np.array(batch_data)[:, :, :, np.newaxis]
        batch_data = np.asarray(batch_data)
        labels += [file_to_label[fpath] for fpath in batch_files]
        current_features = model.predict(batch_data)
        if features is None:
            features = current_features
        else:
            features = np.concatenate((features, current_features))
    return features,labels

def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

def visualize_tsne_dots(tsne,save):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]


    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately

    colors = cycle(
        ['aqua', 'darkorange', 'cornflowerblue', 'yellow', 'red', 'blue', "violet", "teal", "greenyellow", "coral",
         "palegreen", "slateblue","black","indigo","silver","purple","darkgreen","green","lavender"])
    for i, color in zip(range(len(label)), colors):
        # for label in colors_per_class:
        # find the samples of the current class in the data

        indices = [j for j, l in enumerate(labels) if l == label[i]]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)


        # convert the class color to matplotlib format
       # R = random.randint(0, 255)
       # G = random.randint(0, 255)
       # B = random.randint(0, 255)
       # color = np.array([[R, G, B][::-1]], dtype=np.float) / 255
        marker = ['o','x','s']
        m = i % 3
        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, marker=marker[m],c=color, label=label[i])

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    plt.savefig(save)
    plt.close()

# In[2]:
if __name__ == '__main__':


    # In[2]:
    dir = ARGS.res_dir
    tsne_save = "{}".format(dir) + ARGS.freq_compress + "_TSNE.PNG"

    data_dir = ARGS.data_dir
    files, file_to_label = findcsv("test", data_dir)
    files2, file_to_label2 = findcsv("train", data_dir)
    files3, file_to_label3 = findcsv("val", data_dir)
    files = files + files2 + files3
    file_to_label.update(file_to_label2)
    file_to_label.update(file_to_label3)
    folder = ARGS.res_dir

    #model = Resnet18.ResnetBuilder.build_resnet_18((128, 256, 1), 1)
    f = Path(folder + "model_structure.json")
    model_structure = f.read_text()

    # Recreate the Keras model object from the json data
    ae = model_from_json(model_structure)

    # Re-load the model's trained weights
    ae.load_weights(folder + "best_model.h5")
    y = ae.get_layer('encoder').get_output_at(-1)
    y = ae.get_layer('bottleneck').layers[0](y)
    y = ae.get_layer('bottleneck').layers[1](y)
    print(ae.get_layer('bottleneck').layers[1].name)
    y = ae.get_layer('bottleneck').layers[3](y)

    #block_shape = K.int_shape(y)
    #pool2 = AveragePooling2D(pool_size=(block_shape[1], block_shape[2]),
     #                        strides=(1, 1))(y)

    #flatten1 = Flatten()(pool2)
    y = Flatten()(y)

    model = Model(inputs=ae.input, outputs=y)
    model.summary()
    label = getUniqueLabels(data_dir)


    features, labels = getFeatures(files,model)
    print(features.shape)
    print(features)
    #print("features shape ", features.shape)
    tsne = TSNE(n_components=2).fit_transform(features)#p.reshape(features,(-1,1))

    visualize_tsne_dots(tsne,tsne_save)
