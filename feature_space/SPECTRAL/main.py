import sys
sys.path.append("..")

from ploter import *
from sklearn.cluster import KMeans

from Train_Generator import *
from OrganizeData import *

import argparse
import math
import random
from tqdm import tqdm
import numpy as np
from pathlib import Path
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import pandas as pd
import os


from keras.models import model_from_json
from keras.layers import (
    Flatten,
    Dense,
)
from keras import Model



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

ap.add_argument(
    "-nc",
    "--n_clusters",
    type=int,
    default=12,
    help="number of clusters"
)

ARGS = ap.parse_args()

def setseed():
    seed = 1
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
        print("current features shape ", current_features.shape)
        if features is None:
            features = current_features
        else:
            features = np.concatenate((features, current_features))
    return features,labels

def generate_graph_laplacian(df, nn):
    """Generate graph Laplacian from data."""
    # Adjacency Matrix.
    connectivity = kneighbors_graph(X=df, n_neighbors=nn, mode='connectivity')
    adjacency_matrix_s = (1 / 2) * (connectivity + connectivity.T)
    # Graph Laplacian.
    graph_laplacian_s = sparse.csgraph.laplacian(csgraph=adjacency_matrix_s, normed=False)
    graph_laplacian = graph_laplacian_s.toarray()
    return graph_laplacian


def compute_spectrum_graph_laplacian(graph_laplacian):
    """Compute eigenvalues and eigenvectors and project
    them onto the real numbers.
    """
    eigenvals, eigenvcts = np.linalg.eig(graph_laplacian)
    eigenvals = np.real(eigenvals)
    eigenvcts = np.real(eigenvcts)
    return eigenvals, eigenvcts


def project_and_transpose(eigenvals, eigenvcts, num_ev):
    """Select the eigenvectors corresponding to the first
    (sorted) num_ev eigenvalues as columns in a data frame.
    """
    eigenvals_sorted_indices = np.argsort(eigenvals)
    indices = eigenvals_sorted_indices[: num_ev]

    proj_df = pd.DataFrame(eigenvcts[:, indices.squeeze()])
    proj_df.columns = ['v_' + str(c) for c in proj_df.columns]
    return proj_df


def run_k_means(df, n_clusters):
    """K-means clustering."""
    k_means = KMeans(random_state=25, n_clusters=n_clusters)
    k_means.fit(df)
    cluster = k_means.predict(df)
    return cluster

def SpectralClustering(features,n_clusters):
    graph_laplacian = generate_graph_laplacian(df=features, nn=8)
    eigenvals, eigenvcts = compute_spectrum_graph_laplacian(graph_laplacian)
    proj_df = project_and_transpose(eigenvals, eigenvcts, n_clusters)
    clusters = run_k_means(proj_df, proj_df.columns.size)
    return clusters, eigenvals, eigenvcts

def getModel(folder):
    f = Path(folder + "model_structure.json")
    model_structure = f.read_text()

    # Recreate the Keras model object from the json data
    ae = model_from_json(model_structure)

    # Re-load the model's trained weights
    ae.load_weights(folder + "best_model.h5")

    y = ae.get_layer('encoder').get_output_at(-1)
    y = ae.get_layer('bottleneck').layers[0](y) #input
    y = ae.get_layer('bottleneck').layers[1](y) # Conv (1,1)
    y = ae.get_layer('bottleneck').layers[2](y) # batchnormalization
    y = ae.get_layer('bottleneck').layers[3](y) # Relu
    print("y shape before flatten ", y.shape)

    y = Flatten()(y)

    return Model(inputs=ae.input, outputs=y)

def createdict(c_dict,cluster,lab,n_clusters):
    #print("c dict: {}, cluster: {}, lab: {} ,n_clusters: {}".format(c_dict,cluster,lab,n_clusters))
    print("cluster ", cluster)
    for c in range(0,n_clusters):
        if cluster == c:
            if cluster not in c_dict.keys():
                dict = {lab:1}
                c_dict.update({cluster:dict})
            else:
                dict = c_dict.get(cluster)
                if lab not in dict:
                    dict.update({lab:1})
                else:
                    dict[lab] += 1
    return c_dict

def createPiePlots(dict,path):
    for c in dict.keys():
        print("cluster ", c)
        print("c keys:{}, values:{} ".format(dict[c].keys(), dict[c].values()))
        plotPieChart(dict[c].keys(), dict[c].values(), path  + str(c) + ".PNG",c)

if __name__ == '__main__':


    # In[2]:
    dir = ARGS.res_dir
    if not os.path.isdir(dir + "PLOTS/"):
        os.mkdir(dir + "PLOTS/")

    save = dir + "PLOTS/"

    data_dir = ARGS.data_dir
    files, file_to_label = findcsv("test", data_dir)
    files2, file_to_label2 = findcsv("train", data_dir)
    files3, file_to_label3 = findcsv("val", data_dir)
    files = files + files2 + files3
    file_to_label.update(file_to_label2)
    file_to_label.update(file_to_label3)
    folder = ARGS.res_dir

    model = getModel(folder)
    model.summary()


    #label = getUniqueLabels(data_dir)

    features, labels = getFeatures(files,model)
    print("features shape ", features.shape)
    n_clusters=ARGS.n_clusters

    clusters,eigenvals,eigenvcts = SpectralClustering(features,n_clusters)

    plotSortedEigenvalGraphLap(eigenvals, eigenvcts,save)

    plotInertia(features,1,30,save)

    pure_kmeans = KMeans(n_clusters=n_clusters).fit(features.astype('float64'))

    plot(features, clusters, pure_kmeans.labels_, save + "clusters.PNG")

    df = pd.DataFrame(files, columns=["file_name"])
    df['label'] = list(file_to_label.values())
    df['spectral_c'] = clusters
    df['kmeans_c'] = pure_kmeans.labels_
    df.to_csv(folder + "clusters.csv", index=False)

    dfr = pd.read_csv(folder + "clusters.csv")

    spec = {}
    kmeans= {}
    for _,val in enumerate(dfr.values):
        lab = val[1]
        spec_c = val[2]
        kmeans_c = val[3]

        spec = createdict(spec,spec_c,lab,n_clusters)
        kmeans = createdict(kmeans,kmeans_c,lab,n_clusters)

    if not os.path.isdir(dir + "PLOTS/SPEC/"):
        os.mkdir(dir + "PLOTS/SPEC/")
    spec_path = dir + "PLOTS/SPEC/"
    createPiePlots(spec,spec_path)

    if not os.path.isdir(dir + "PLOTS/KMEANS/"):
        os.mkdir(dir + "PLOTS/KMEANS/")
    kmeans_path = dir + "PLOTS/KMEANS/"
    createPiePlots(kmeans,kmeans_path)







