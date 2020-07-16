from matplotlib import pyplot as plt
from itertools import cycle, islice
import numpy as np
import seaborn as sns
sns.set_style('darkgrid', {'axes.facecolor': '.9'})
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')
from sklearn.cluster import KMeans


def plot(X, y_sp, y_km,path):
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                '#f781bf', '#a65628', '#984ea3',
                                                '#999999', '#e41a1c', '#dede00']),
                                        int(max(y_km) + 1))))

    print("X ", X)
    plt.subplot(121)
    plt.scatter(np.real(X[:,0]), np.real(X[:,1]), s=10, color=colors[y_sp])
    plt.scatter(np.imag(X[:,0]), np.imag(X[:,1]), s=10, color=colors[y_sp])
    plt.title("Spectral Clustering")
    plt.subplot(122)
    plt.scatter(np.real(X[:,0]), np.real(X[:,1]), s=10, color=colors[y_km])
    plt.scatter(np.imag(X[:, 0]), np.imag(X[:, 1]), s=10, color=colors[y_km])
    plt.title("Kmeans Clustering")
    # plt.show()
    plt.savefig(path)

def plotSortedEigenvalGraphLap(eigenvals, eigenvcts,path):
    eigenvcts_norms = np.apply_along_axis(
        lambda v: np.linalg.norm(v, ord=2),
        axis=0,
        arr=eigenvcts
    )

    print('Min Norm: ' + str(eigenvcts_norms.min()))
    print('Max Norm: ' + str(eigenvcts_norms.max()))

    eigenvals_sorted_indices = np.argsort(eigenvals)
    eigenvals_sorted = eigenvals[eigenvals_sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(x=range(1, eigenvals_sorted_indices.size + 1), y=eigenvals_sorted, ax=ax)
    ax.set(title='Sorted Eigenvalues Graph Laplacian', xlabel='index', ylabel=r'$\lambda$');
    fig.savefig(path + "SortedEV.PNG")

    plt.close()

    index_lim = 10

    figx, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=range(1, eigenvals_sorted_indices[: index_lim].size + 1), y=eigenvals_sorted[: index_lim], s=80,
                    ax=ax)
    sns.lineplot(x=range(1, eigenvals_sorted_indices[: index_lim].size + 1), y=eigenvals_sorted[: index_lim], alpha=0.5,
                 ax=ax)
    #ax.axvline(x=3, color=sns_c[3], label='zero eigenvalues', linestyle='--')
    ax.legend()
    ax.set(title=f'Sorted Eigenvalues Graph Laplacian (First {index_lim})', xlabel='index', ylabel=r'$\lambda$');
    figx.savefig(path + "Zoomed.PNG")
    plt.close()

def plotInertia(features,min,max,path):
    k_candidates = range(min, max)

    inertias = []
    for k in k_candidates:
        k_means = KMeans(random_state=42, n_clusters=k)
        k_means.fit(features)
        inertias.append(k_means.inertia_)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=k_candidates, y=inertias, s=80, ax=ax)
    # sns.scatterplot(x=[k_candidates[3]], y=[inertias[2]], color=sns_c[3], s=150, ax=ax)
    sns.lineplot(x=k_candidates, y=inertias, alpha=0.5, ax=ax)
    ax.set(title='Inertia K-Means', ylabel='inertia', xlabel='k');
    fig.savefig(path + "inertia.PNG")
    plt.close()

def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d})".format(pct, absolute)

def plotPieChart(labels,sizes,path,c):
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:

    fig1, ax1 = plt.subplots()
    _, texts, autotexts = ax1.pie(list(sizes), labels=labels, autopct=lambda pct: func(pct, list(sizes)),
                shadow=False, startangle=90)
    plt.setp(autotexts, size=8, weight="bold")
    ax1.set_title("Cluster {}".format(c))
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig(path)
    plt.close()

