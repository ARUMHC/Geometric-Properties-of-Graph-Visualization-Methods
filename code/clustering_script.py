import pandas as pd
import networkx as nx
from sklearn.cluster import Birch
import numpy as np
from networkx.algorithms.community.centrality import girvan_newman
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans, DBSCAN, Birch
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')
import igraph as ig
import community
from graph_generating_script import *


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

def find_best_num_clusters(data, max_clusters=10):
    best_score = -1
    best_num_clusters = 2  # Minimum number of clusters
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_num_clusters = n_clusters
    
    return best_num_clusters


from sklearn.cluster import AgglomerativeClustering, OPTICS


def get_clusters_from_positions(G, posdf, best_num, true_labels, k=1):
    #agglomerative clustering
    scores = []
    # for i in range(k):
    model = AgglomerativeClustering(affinity='euclidean', n_clusters=best_num)
    model.fit(posdf)
    yhat = list(model.labels_)
    # scores[0] += adjusted_rand_score(true_labels, yhat)
    scores.append(adjusted_rand_score(true_labels, yhat))

    #DBscan
    model = OPTICS()
    model.fit(posdf)
    yhat= list(model.labels_)
    # scores[1] += adjusted_rand_score(true_labels, yhat)
    scores.append(adjusted_rand_score(true_labels, yhat))

    # kmeans
    model = KMeans(n_clusters=best_num, random_state=212)
    model.fit(posdf)
    yhat = list(model.predict(posdf))
    # scores[2] += adjusted_rand_score(true_labels, yhat)
    scores.append(adjusted_rand_score(true_labels, yhat))

    #GMM
    from sklearn.mixture import GaussianMixture
    #modeling
    model = GaussianMixture(n_components=best_num, random_state=212).fit(posdf)
    yhat = list(model.predict(posdf))
    # scores[3] += adjusted_rand_score(true_labels, yhat)
    scores.append(adjusted_rand_score(true_labels, yhat))

    #Birch
    model = Birch(n_clusters=best_num)
    model.fit(posdf)
    yhat = list(model.predict(posdf))
    # scores[4] += adjusted_rand_score(true_labels, yhat)
    scores.append(adjusted_rand_score(true_labels, yhat))

    # communities = girvan_newman(G)

    #community detections

    #Girvan Newman
    # communities = girvan_newman(G)
    # comms = []
    # for com in next(communities):
    #     comms.append(list(com))
    # list_comms = [None] * len(G.nodes)
    # for i in range(len(comms)):
    #     com = comms[i]
    #     for node in com:
    #         list_comms[node] = i

    #separate tool to choose number of communities for girvan newman
    partition = community.best_partition(G)
    num_communities = len(set(partition.values()))
    # print(f'Number for communities for Girvan Newman: {num_communities}')
    G1 = G.copy()
    
    while nx.number_connected_components(G1) < num_communities:
        edge_centrality = nx.edge_betweenness_centrality(G1)
        max_edge = max(edge_centrality, key=edge_centrality.get)
        G1.remove_edge(*max_edge)
    
    communities = list(nx.connected_components(G1))
    comms = []
    for com in communities:
        comms.append(list(com))
    list_comms = [None] * len(G.nodes)
    for i in range(len(comms)):
        com = comms[i]
        for node in com:
            list_comms[node] = i

    scores.append(adjusted_rand_score(true_labels, list_comms))
    # scores[5] += adjusted_rand_score(true_labels, list_comms)

    #Leiden
    G_ig = ig.Graph.TupleList(nx.to_edgelist(G), directed=False)
    partition = G_ig.community_leiden(objective_function="modularity", resolution=1)
    list_comms = [None] * len(G.nodes)
    for i, com in enumerate(partition):
        for node in com:
            list_comms[node] = i

    # scores[6] += adjusted_rand_score(true_labels, list_comms)
    scores.append(adjusted_rand_score(true_labels, list_comms))
    return scores

def add_scores(df, scores, layout_name='spring'):
    scores.insert(0, layout_name)
    data = df.to_dict('records')
    data.append(dict(zip(df.columns, scores)))
    df = pd.DataFrame(data)
    return df

def scaling_igraph(layout):
    coords = np.array(layout.coords)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    scaled_coords = 2 * (coords - min_coords) / (max_coords - min_coords) - 1
    posdf = pd.DataFrame(scaled_coords, columns=['X', 'Y'])
    return posdf

def full_experiment_on_graph(G, true_labels):
    df = pd.DataFrame(columns=['layout','AgglomerativeClustering', 'OPTICS', 'KMeans', 'GMM', 'Birch', 'Girvan Newman', 'Leiden'])
    
    #for every layout
    #kamada kawai
    pos = nx.kamada_kawai_layout(G)
    posdf = pd.DataFrame.from_dict(pos, orient='index', columns=['X', 'Y'])
    best_num = find_best_num_clusters(posdf)
    # print(f'Best number of clusters detected : {best_num}')
    scores = get_clusters_from_positions(G,posdf, best_num, true_labels)
    df = add_scores(df, scores, 'kamada_kawai')

    #spring layout
    pos = nx.spring_layout(G)
    posdf = pd.DataFrame.from_dict(pos, orient='index', columns=['X', 'Y'])
    best_num = find_best_num_clusters(posdf)
    scores = get_clusters_from_positions(G,posdf, best_num, true_labels)
    df = add_scores(df, scores, 'spring')

    #algorithms from igraph
    G_ig = ig.Graph.TupleList(nx.to_edgelist(G), directed=False)

    #davidson harel
    layout = G_ig.layout('davidson_harel')
    posdf = scaling_igraph(layout)
    # posdf = pd.DataFrame(layout.coords, columns=['X', 'Y'])
    best_num = find_best_num_clusters(posdf)
    scores = get_clusters_from_positions(G, posdf, best_num, true_labels)
    df = add_scores(df, scores, 'davidson_harel')

    #drl
    layout = G_ig.layout('drl')
    posdf = scaling_igraph(layout)
    # posdf = pd.DataFrame(layout.coords, columns=['X', 'Y'])
    best_num = find_best_num_clusters(posdf)
    scores = get_clusters_from_positions(G, posdf, best_num, true_labels)
    df = add_scores(df, scores, 'drl')

    # fruchterman reingold
    layout = G_ig.layout('fruchterman_reingold')
    posdf = scaling_igraph(layout)
    # posdf = pd.DataFrame(layout.coords, columns=['X', 'Y'])
    best_num = find_best_num_clusters(posdf)
    scores = get_clusters_from_positions(G, posdf, best_num, true_labels)
    df = add_scores(df, scores, 'fruchterman_reingold')

    #graphopt
    layout = G_ig.layout('graphopt')
    posdf = scaling_igraph(layout)
    # posdf = pd.DataFrame(layout.coords, columns=['X', 'Y'])
    best_num = find_best_num_clusters(posdf)
    scores = get_clusters_from_positions(G, posdf, best_num, true_labels)
    df = add_scores(df, scores, 'graphopt')

    #lgl
    layout = G_ig.layout('lgl')
    posdf = scaling_igraph(layout)
    # posdf = pd.DataFrame(layout.coords, columns=['X', 'Y'])
    best_num = find_best_num_clusters(posdf)
    scores = get_clusters_from_positions(G,  posdf, best_num, true_labels)
    df = add_scores(df, scores, 'lgl')

    #mds
    layout = G_ig.layout('mds')
    posdf = scaling_igraph(layout)
    # posdf = pd.DataFrame(layout.coords, columns=['X', 'Y'])
    best_num = find_best_num_clusters(posdf)
    scores = get_clusters_from_positions(G, posdf, best_num, true_labels)
    df = add_scores(df, scores, 'mds')

    return df
            
def steady_full_experiment(sizes, inside_prob, outside_prob, k=5):

    (G, true_labels)= generate_G(sizes, inside_prob, outside_prob)
    df = full_experiment_on_graph(G, true_labels)
    # print("0")
    # print(df)
    for i in range(1,k):
        (G, true_labels)= generate_G(sizes, inside_prob, outside_prob)
        tmp = full_experiment_on_graph(G, true_labels)
        df[df.columns[1:]] += tmp[tmp.columns[1:]]

    df[df.columns[1:]] /= k
    return df
    