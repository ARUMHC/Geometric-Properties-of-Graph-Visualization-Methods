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
from sklearn.model_selection import ParameterGrid
import networkx as nx


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np



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


# get results from one model on one graph on one of the layouts
# returns : scores - list with ARI

def get_clusters_from_positions(posdf, best_num, true_labels):
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


    return scores


def get_communities(G, true_labels):
    scores = []
    #separate tool to choose number of communities for girvan newman
    partition = community.best_partition(G)
    num_communities = len(set(partition.values()))
    # print(f'community number {num_communities}')
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
    # G_ig = ig.Graph.TupleList(nx.to_edgelist(G), directed=False)
    # partition = G_ig.community_leiden(objective_function="modularity")
    # list_comms = [None] * len(G.nodes)
    # for i, com in enumerate(partition):
    #     for node in com:
    #         list_comms[node] = i
    G_ig = ig.Graph.TupleList(nx.to_edgelist(G), directed=False)

    resolutions = np.linspace(0.1, 1.5, 10)  # Adjust the range as needed
    param_grid = {'resolution': resolutions}
    grid = ParameterGrid(param_grid)

    best_modularity = -np.inf
    best_partition = None
    for params in grid:
        partition = G_ig.community_leiden(objective_function="modularity", **params)
        modularity = G_ig.modularity(partition)
        if modularity > best_modularity:
            best_modularity = modularity
            best_partition = partition

    list_comms = [None] * len(G.nodes)
    for i, com in enumerate(best_partition):
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

#gap statistics
def gap_num_clusters(data, nrefs=5, maxClusters=7):
    """
    Calculates KMeans optimal K using Gap Statistic 
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    # resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters)):
        refDisps = np.zeros(nrefs)
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp
        km = KMeans(k)
        km.fit(data)
        origDisp = km.inertia_
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
        gaps[gap_index] = gap
                
    return gaps.argmax() + 1


# def scaling_igraph(layout):
#     coords = np.array(layout.coords)
#     min_coords = coords.min(axis=0)
#     max_coords = coords.max(axis=0)
#     scaled_coords = 2 * (coords - min_coords) / (max_coords - min_coords) - 1
#     posdf = pd.DataFrame(scaled_coords, columns=['X', 'Y'])
#     return posdf


# coducts ONE experiemnt for all (7) the layouts
# returns : df with ARI layouts and algoriths for ONE graph

def full_cluster_experiment(G, true_labels, silhuoette=False):
    # df = pd.DataFrame(columns=['layout','AgglomerativeClustering', 'OPTICS', 'KMeans', 'GMM', 'Birch', 'Girvan Newman', 'Leiden'])
    df = pd.DataFrame(columns=['layout','AgglomerativeClustering', 'OPTICS', 'KMeans', 'GMM', 'Birch'])

    #for every layout
    #kamada kawai
    pos = nx.kamada_kawai_layout(G)
    posdf = pd.DataFrame.from_dict(pos, orient='index', columns=['X', 'Y'])
    best_num = find_best_num_clusters(posdf) if silhuoette else gap_num_clusters(posdf)
  
    # print(f'Best number of clusters detected : {best_num}')
    scores = get_clusters_from_positions(posdf, best_num, true_labels)
    df = add_scores(df, scores, 'kamada_kawai')

    #spring layout
    pos = nx.spring_layout(G)
    posdf = pd.DataFrame.from_dict(pos, orient='index', columns=['X', 'Y'])
    # best_num = find_best_num_clusters(posdf)
    best_num = find_best_num_clusters(posdf) if silhuoette else gap_num_clusters(posdf)

    scores = get_clusters_from_positions(posdf, best_num, true_labels)
    df = add_scores(df, scores, 'spring')

    #algorithms from igraph
    G_ig = ig.Graph.TupleList(nx.to_edgelist(G), directed=False)

    #davidson harel
    layout = G_ig.layout('davidson_harel')
    # posdf = scaling_igraph(layout)
    posdf = pd.DataFrame(layout.coords, columns=['X', 'Y'])
    # best_num = find_best_num_clusters(posdf)
    best_num = find_best_num_clusters(posdf) if silhuoette else gap_num_clusters(posdf)


    scores = get_clusters_from_positions( posdf, best_num, true_labels)
    df = add_scores(df, scores, 'davidson_harel')

    #drl
    layout = G_ig.layout('drl')
    # posdf = scaling_igraph(layout)
    posdf = pd.DataFrame(layout.coords, columns=['X', 'Y'])
    # best_num = find_best_num_clusters(posdf)
    best_num = find_best_num_clusters(posdf) if silhuoette else gap_num_clusters(posdf)

    scores = get_clusters_from_positions( posdf, best_num, true_labels)
    df = add_scores(df, scores, 'drl')

    # fruchterman reingold
    layout = G_ig.layout('fruchterman_reingold')
    # posdf = scaling_igraph(layout)
    posdf = pd.DataFrame(layout.coords, columns=['X', 'Y'])
    # best_num = find_best_num_clusters(posdf)
    best_num = find_best_num_clusters(posdf) if silhuoette else gap_num_clusters(posdf)

    scores = get_clusters_from_positions( posdf, best_num, true_labels)
    df = add_scores(df, scores, 'fruchterman_reingold')

    #graphopt
    layout = G_ig.layout('graphopt')
    # posdf = scaling_igraph(layout)
    posdf = pd.DataFrame(layout.coords, columns=['X', 'Y'])
    # best_num = find_best_num_clusters(posdf)
    best_num = find_best_num_clusters(posdf) if silhuoette else gap_num_clusters(posdf)

    scores = get_clusters_from_positions( posdf, best_num, true_labels)
    df = add_scores(df, scores, 'graphopt')

    #lgl
    layout = G_ig.layout('lgl')
    # posdf = scaling_igraph(layout)
    posdf = pd.DataFrame(layout.coords, columns=['X', 'Y'])
    # best_num = find_best_num_clusters(posdf)
    best_num = find_best_num_clusters(posdf) if silhuoette else gap_num_clusters(posdf)
    scores = get_clusters_from_positions( posdf, best_num, true_labels)
    df = add_scores(df, scores, 'lgl')

    #mds
    layout = G_ig.layout('mds')
    # posdf = scaling_igraph(layout)
    posdf = pd.DataFrame(layout.coords, columns=['X', 'Y'])
    # best_num = find_best_num_clusters(posdf)
    best_num = find_best_num_clusters(posdf) if silhuoette else gap_num_clusters(posdf)


    scores = get_clusters_from_positions(posdf, best_num, true_labels)
    df = add_scores(df, scores, 'mds')

    return df

# generates k graphs and conducts FULL experiments on them
# it sums up ARIs and divides by k (average)
# returns : df


def steady_full_experiment(n_vertex, n_comms, inside_prob, outside_prob, k=5, i_want_boxplot=False, dispersion=.35):
    # (G, true_labels)= generate_G(sizes, inside_prob, outside_prob)
    print('0')
    (G, true_labels) = generate_G_randomized(n_vertex, n_comms, inside_prob, outside_prob)
    asor = nx.numeric_assortativity_coefficient(G, "community")
    df = full_cluster_experiment(G, true_labels)
    df.iloc[:, 1:] = df.iloc[:, 1:].applymap(lambda x: [x])
    #comms - list with two values
    comms = get_communities(G, true_labels)
    girvs = [comms[0]]
    leid = [comms[1]]
    #tu trzeba rozbic i osobno zrobic clustry, osobno community detection
    for i in range(1, k):
        print(i)
        # (G, true_labels)= generate_G(sizes, inside_prob, outside_prob)
        (G, true_labels) = generate_G_randomized(n_vertex, n_comms, inside_prob, outside_prob)
        asor += nx.numeric_assortativity_coefficient(G, "community")
        tmp = full_cluster_experiment(G, true_labels)
        # df[df.columns[1:]] += tmp[tmp.columns[1:]]
        for column in df.columns[1:]:
            df[column] = df[column].combine(tmp[column], lambda x, y: x + [y])
            
        tmp = get_communities(G, true_labels)
        girvs.append(tmp[0])
        leid.append(tmp[1])
        # comms[0] += tmp[0]
        # comms[1] += tmp[1]


    # df[df.columns[1:]] /= k
    # df['Girvan-Newman'] = comms[0]/k
    df['Girvan-Newman'] = [girvs] * len(df)
    # df['Leiden'] = comms[1]/k
    df['Leiden'] = [leid] * len(df)

    print(f'Graphs assortavity coefficient : {asor/k}')
    if i_want_boxplot==False:
        df.iloc[:, 1:] = df.iloc[:, 1:].applymap(lambda x: sum(x) / k)

    return df
    
    
def generate_G(sizes, inside_prob, outside_prob):
    probs = np.eye(len(sizes)) * inside_prob

    # Set the off-diagonal elements to the desired value (0.01)
    probs[probs == 0] = outside_prob
    true_labels=[]
    i=0
    for size in sizes:
        true_labels += ([i]*size)
        i += 1
    G = nx.stochastic_block_model(sizes, probs, seed=random.randint(0, 200))
    for node, community in zip(G.nodes(), true_labels):
        G.nodes[node]['community'] = community
        
    return (G, true_labels)