# (c) Grigory Syrabriakov https://www.learnopencv.com/t-sne-for-feature-visualization/
# In[0]:
from Train_Generator import *
from OrganizeData import *

from sklearn.cluster import KMeans, SpectralClustering
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


class TSNE_plotter():

    def __init__(self):
        self.setseed()

    def setseed(self):
        seed = 10
        random.seed(seed)
        np.random.seed(seed)

    def getFeatures(self, files, model):

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
        return features, labels

    def scale_to_01_range(self, x):
        # compute the distribution range
        value_range = (np.max(x) - np.min(x))

        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)

        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range

    def visualize_tsne_dots(self, tsne, save,label,labels):
        # extract x and y coordinates representing the positions of the images on T-SNE plot
        tx = tsne[:, 0]
        ty = tsne[:, 1]

        tx = self.scale_to_01_range(tx)
        ty = self.scale_to_01_range(ty)

        # initialize a matplotlib plot
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # for every class, we'll add a scatter plot separately

        colors = cycle(
            ['aqua', 'darkorange', 'cornflowerblue', 'yellow', 'red', 'blue', "violet", "teal", "greenyellow", "coral",
             "darkgreen", "slateblue", "black", "indigo", "silver", "purple", "palegreen", "green", "bittersweet"])
        for i, color in zip(range(len(label)), colors):
            # for label in colors_per_class:
            # find the samples of the current class in the data

            indices = [j for j, l in enumerate(labels) if l == label[i]]

            # extract the coordinates of the points of this class only
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            marker = ['o', 'x', 's']
            m = i % 3
            # add a scatter plot with the corresponding color and label
            ax.scatter(current_tx, current_ty, marker=marker[m], c=color, label=label[i])

        # build a legend using the labels we set previously
        ax.legend(loc='best')

        # finally, show the plot
        plt.savefig(save)
        plt.close()
        return tx,ty


def getModel(path):
    f = Path(folder + "model_structure.json")
    model_structure = f.read_text()

    # Recreate the Keras model object from the json data
    ae = model_from_json(model_structure)

    # Re-load the model's trained weights
    ae.load_weights(folder + "best_model.h5")
    # y = ae.layers[-2].output
    y = ae.get_layer('encoder').get_output_at(-1)
    y = ae.get_layer('bottleneck').layers[0](y)
    y = ae.get_layer('bottleneck').layers[1](y)
    # print(ae.get_layer('bottleneck').layers[1].name)
    y = ae.get_layer('bottleneck').layers[3](y)

    # block_shape = K.int_shape(y)
    # pool2 = AveragePooling2D(pool_size=(block_shape[1], block_shape[2]),
    #                        strides=(1, 1))(y)

    # flatten1 = Flatten()(pool2)
    y = Flatten()(y)

    model = Model(inputs=ae.input, outputs=y)
    model.summary()
    return model


# In[2]:
if __name__ == '__main__':

    # In[2]:
    dir = ARGS.res_dir

    data_dir = ARGS.data_dir
    files, file_to_label = findcsv("unseen", data_dir)
    # files, file_to_label = findcsv("test", data_dir)
    # files2, file_to_label2 = findcsv("train", data_dir)
    # files3, file_to_label3 = findcsv("val", data_dir)
    # files = files + files2 + files3
    # file_to_label.update(file_to_label2)
    # file_to_label.update(file_to_label3)
    folder = ARGS.res_dir
    model = getModel(folder)
    # model = Resnet18.ResnetBuilder.build_resnet_18((128, 256, 1), 1)
    tsne_p = TSNE_plotter()

    label = getUniqueLabels(data_dir)
    features, labels = getFeatures(files, model)
    print(features.shape)
    print(features)
    # print("features shape ", features.shape)
    # p = 1
    # for i in range(10):
    #    print("ROUND ",i)

    for x in range(50):
        print("ROUND ", x)
        tsne_save = "{}".format(dir) + ARGS.freq_compress + "_TSNE_{}.PNG".format(x)
        tsne = TSNE(n_components=2, perplexity=34, n_iter=5000, learning_rate=50, verbose=2,
                    early_exaggeration=4.0).fit_transform(features)  # p.reshape(features,(-1,1))
        # print("kl divergence score ", tsne.kl_divergence_)
        tsne_p.visualize_tsne_dots(tsne, tsne_save)
        # p += 1
