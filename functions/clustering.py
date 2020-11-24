#! /usr/bin/env python3
# coding: utf-8

""" This module gathers all functions required for data clustering
with machine learning algorithms."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def save_fig(fig_id, tight_layout=True):
    folder_path = os.path.join("charts")
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    path = os.path.join("charts", fig_id + ".png")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def kmeans_scores_plot(X, file_name="scores_vs_k_plot"):
    """Plot both the Silhouette Score et R² score for different values of k
    -----------
    Parameters :
    X: Array
        the array object holding data
    file_name : String
        name of the png file
    -----------
    Return :
    axes : Matplotlib axis object
    """
    # Get the models for different k
    kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X_scaled)
                    for k in range(2, 11)]
    # Compute the R2 score
    total_inertia = np.var(X_scaled, axis=0).sum()
    intra_inertia = [model.inertia_/len(X_scaled)
                     for model in kmeans_per_k]
    r2_scores = [1 - inertia/total_inertia
                 for inertia in intra_inertia]

    # Compute the silhouette score
    sample_size = int(X_scaled.shape[0]*0.2)
    s_scores = [silhouette_score(X_scaled, model.labels_,
                                 metric='euclidean',
                                 sample_size=sample_size,
                                 random_state=42)
                for model in kmeans_per_k]

    # Create the plot
    plt.plot(range(2, 11), s_scores, "ro-", label="Silhouette Score")
    plt.plot(range(2, 11), r2_scores, "ks-", label="R² Score")
    plt.xlabel("Number of Clusters")
    plt.legend()
    save_fig(file_name)
    plt.show()

def silhouette_diag(X, list_k, file_name="silhouette_diagram_plot"):
    """Plot a silhouette diagram for different values of k
    -----------
    Parameters :
    X: Array
        the array object holding data
    list_k: List
        list with the different values of k
    file_name : String
        name of the png file
    -----------
    Return :
    axes : Matplotlib axis object
    """
    # Get the figure size
    n_k = len(list_k)
    if n_k % 2 == 0:
        n_rows = n_k / 2
    else:
        n_rows = n_k // 2 + 1
    fig = plt.figure(figsize=(10, n_rows*4))

    # Get the models for different k
    kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)
                    for k in list_k]
    # Get the silhouette scores
    sample_size = int(X.shape[0]*0.1)
    s_scores = [silhouette_score(X, model.labels_,
                                 metric='euclidean',
                                 sample_size=sample_size,
                                 random_state=42)
                for model in kmeans_per_k]
    # Plot the silhouette diagram
    for i, k in enumerate(list_k):
        plt.subplot(2, 2, i + 1)
        y_pred = kmeans_per_k[i].labels_
        silhouette_coefficients = silhouette_samples(X, y_pred)
        padding = len(X) // 30
        pos = padding
        ticks = []
        for j in range(k):
            coeffs = silhouette_coefficients[y_pred == j]
            coeffs.sort()
            color = mpl.cm.Spectral(j / k)
            plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                              facecolor=color, edgecolor=color, alpha=0.7)
            ticks.append(pos + len(coeffs) // 2)
            pos += len(coeffs) + padding

        plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
        plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
        plt.gca().set_xticks(np.linspace(-0.2, 1, 7))
        if i % 2 == 0:
            plt.ylabel("Cluster")
        if i >= (n_rows - 1) * 2:
            plt.xlabel("Silhouette Coefficient")
        else:
            plt.tick_params(labelbottom=False)
        plt.axvline(x=s_scores[k - 2], color="k", linestyle="--")
        plt.title("$k={}$".format(k))

    save_fig(file_name)
    plt.show()
