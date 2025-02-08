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
import community.community_louvain as community
from graph_generating_script import *
from sklearn.model_selection import ParameterGrid
import networkx as nx
import genieclust

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from tqdm import tqdm

from sklearn.cluster import AgglomerativeClustering, OPTICS

# from code.NUM_CLUST_choosing_method.clus_num_methods import mix_ch_elbow
from NUM_CLUST_choosing_method.clus_num_methods import mix_ch_elbow

# get results from one model on one graph on one of the layouts
# returns : scores - list with ARI

#done refactor it so it returns a dictionary
def get_clustering_scores_from_positions(posdf, best_num, true_labels)->dict:

    '''
    get results from one model on one graph on one of the layouts
    returns : scores - list with ARI
    here lies training of clustering models

    ALL CLUSTERING MODELS FOR ONE LAYOUT
    
    returns
    dict with cluster algorithm name : ARI score
    '''
    #agglomerative clustering
    ari_scores = {}
    model = AgglomerativeClustering(n_clusters=best_num)
    try:
        model.fit(posdf)
        yhat = list(model.labels_)
        ari_scores['AgglomerativeClustering'] = adjusted_rand_score(true_labels, yhat)
    except:
        ari_scores['AgglomerativeClustering'] = 'ERROR'
    

    #DBscan
    model = OPTICS()
    try:
        model.fit(posdf)
        yhat= list(model.labels_)
        ari_scores['OPTICS'] = adjusted_rand_score(true_labels, yhat)
    except:
        ari_scores['OPTICS'] = 'ERROR'
    # kmeans
    model = KMeans(n_clusters=best_num, random_state=212)
    try:
        model.fit(posdf)
        yhat = list(model.predict(posdf))
        ari_scores['KMeans'] = adjusted_rand_score(true_labels, yhat)
    except:
        ari_scores['KMeans'] = 'ERROR'

    #GMM
    #modeling
    model = GaussianMixture(n_components=best_num, random_state=212)
    try:
        model.fit(posdf)
        yhat = list(model.predict(posdf))
        ari_scores['GMM'] = adjusted_rand_score(true_labels, yhat)
    except:
        ari_scores['GMM'] = 'ERROR'
    
    #Birch
    model = Birch(n_clusters=best_num)
    try:
        model.fit(posdf)
        yhat = list(model.predict(posdf))
        ari_scores['Birch'] = adjusted_rand_score(true_labels, yhat)
    except:
        ari_scores['Birch'] = 'ERROR'

    # Genie
    try:
        model = genieclust.Genie(n_clusters=best_num)
        yhat = list(model.fit_predict(posdf))
        ari_scores['Genie'] = adjusted_rand_score(true_labels, yhat)
    except:
        ari_scores['Genie'] = 'ERROR'


    return ari_scores

#done name this better
def get_communities_scores_from_positions(G, true_labels):
    '''
    separate function for community detection
    '''
    ari_scores = {}

    #@ GIRVAN NEWMAN
    #separate tool to choose number of communities for girvan newman
    partition = community.best_partition(G)
    num_communities = len(set(partition.values()))
  
    G1 = G.copy()
    
    while nx.number_connected_components(G1) < num_communities:
        edge_centrality = nx.edge_betweenness_centrality(G1)
        max_edge = max(edge_centrality, key=edge_centrality.get)
        G1.remove_edge(*max_edge)
    
    communities = list(nx.connected_components(G1))
    list_comms = [i for i, com in enumerate(communities) for node in com]

    ari_scores['Girvan Newman'] = adjusted_rand_score(true_labels, list_comms)
    # scores[5] += adjusted_rand_score(true_labels, list_comms)

    #@ LEIDEN
    G_ig = ig.Graph.TupleList(nx.to_edgelist(G), directed=False)
    missing_vertices = set(G.nodes()) - set(G_ig.vs['name'])
    G_ig.add_vertices(list(missing_vertices))

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

    list_comms = [i for i, com in enumerate(best_partition) for node in com]
    ari_scores['Leiden'] = adjusted_rand_score(true_labels, list_comms)

    return ari_scores

def rescale_igraph_pos(pos):
    coords = np.array(list(pos.values()))
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    scaled_coords = 2 * (coords - min_coords) / (max_coords - min_coords) - 1

    # Create a dictionary of rescaled positions for NetworkX
    rescaled_pos = {i: scaled_coords[i] for i in range(len(scaled_coords))}
    return rescaled_pos

def rescale_dataframe_coords(df):
    coords = df.values
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    scaled_coords = 2 * (coords - min_coords) / (max_coords - min_coords) - 1
    
    # Create a new DataFrame with the rescaled coordinates
    rescaled_df = pd.DataFrame(scaled_coords, columns=df.columns, index=df.index)
    return rescaled_df

# def scaling_igraph(layout):
#     coords = np.array(layout.coords)
#     min_coords = coords.min(axis=0)
#     max_coords = coords.max(axis=0)
#     scaled_coords = 2 * (coords - min_coords) / (max_coords - min_coords) - 1
#     posdf = pd.DataFrame(scaled_coords, columns=['X', 'Y'])
#     return posdf


# coducts ONE experiemnt for all (7) the layouts
# returns : df with ARI layouts and algoriths for ONE graph

def posdf_from_layout(G, layout_name):

    if layout_name=='kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
        posdf = pd.DataFrame.from_dict(pos, orient='index', columns=['X', 'Y'])
    elif layout_name == 'spring':
        pos = nx.spring_layout(G)
        posdf = pd.DataFrame.from_dict(pos, orient='index', columns=['X', 'Y'])
    elif layout_name=='davidson_harel':
        G_ig = ig.Graph.TupleList(nx.to_edgelist(G), directed=False)
        missing_vertices = set(G.nodes()) - set(G_ig.vs['name'])
        G_ig.add_vertices(list(missing_vertices))
        layout = G_ig.layout('davidson_harel')
        # layout = rescale_igraph_pos(layout)
        posdf = pd.DataFrame(layout.coords, columns=['X', 'Y'])
        posdf = rescale_dataframe_coords(posdf)
    elif layout_name=='drl':
        G_ig = ig.Graph.TupleList(nx.to_edgelist(G), directed=False)
        missing_vertices = set(G.nodes()) - set(G_ig.vs['name'])
        G_ig.add_vertices(list(missing_vertices))
        layout = G_ig.layout('drl')
        posdf = pd.DataFrame(layout.coords, columns=['X', 'Y'])
        posdf = rescale_dataframe_coords(posdf)
    elif layout_name=='fruchterman_reingold':
        G_ig = ig.Graph.TupleList(nx.to_edgelist(G), directed=False)
        missing_vertices = set(G.nodes()) - set(G_ig.vs['name'])
        G_ig.add_vertices(list(missing_vertices))
        layout = G_ig.layout('fruchterman_reingold')
        posdf = pd.DataFrame(layout.coords, columns=['X', 'Y'])
        posdf = rescale_dataframe_coords(posdf)
    elif layout_name=='graphopt':
        G_ig = ig.Graph.TupleList(nx.to_edgelist(G), directed=False)
        missing_vertices = set(G.nodes()) - set(G_ig.vs['name'])
        G_ig.add_vertices(list(missing_vertices))
        layout = G_ig.layout('graphopt')
        posdf = pd.DataFrame(layout.coords, columns=['X', 'Y'])
        posdf = rescale_dataframe_coords(posdf)
    elif layout_name=='lgl':
        G_ig = ig.Graph.TupleList(nx.to_edgelist(G), directed=False)
        missing_vertices = set(G.nodes()) - set(G_ig.vs['name'])
        G_ig.add_vertices(list(missing_vertices))
        layout = G_ig.layout('lgl')
        posdf = pd.DataFrame(layout.coords, columns=['X', 'Y'])
        posdf = rescale_dataframe_coords(posdf)
    elif layout_name=='mds':
        G_ig = ig.Graph.TupleList(nx.to_edgelist(G), directed=False)
        missing_vertices = set(G.nodes()) - set(G_ig.vs['name'])
        G_ig.add_vertices(list(missing_vertices))
        layout = G_ig.layout('mds')
        posdf = pd.DataFrame(layout.coords, columns=['X', 'Y'])
        posdf = rescale_dataframe_coords(posdf)
    else:
        raise ValueError('Wrong layout name (probably typo)')
    return posdf

def best_number_of_clusters(G, max_clusters=10):
    # posdf = posdf_from_layout(G, 'graphopt')
    G_ig = ig.Graph.TupleList(nx.to_edgelist(G), directed=False)
    missing_vertices = set(G.nodes()) - set(G_ig.vs['name'])
    G_ig.add_vertices(list(missing_vertices))
    layout = G_ig.layout('fruchterman_reingold')
    posdf = pd.DataFrame(layout.coords, columns=['X', 'Y'])
    posdf = rescale_dataframe_coords(posdf)
    best_num = mix_ch_elbow(posdf, .75, .25, max_clusters=max_clusters)

    return best_num


def calculate_scores_for_layout(G, true_labels, layout_name):
    '''
    helper function

    returns
    {'AgglomerativeClustering': 1.0, 'OPTICS': 0.9623418543390346, 'KMeans': 1.0, 'GMM': 1.0, 'Birch': 0.6011004126547456}
  
    '''
    posdf = posdf_from_layout(G, layout_name)
    best_num = best_number_of_clusters(G)
    # best_num = find_best_num_clusters(posdf)
    ari_scores = get_clustering_scores_from_positions(posdf, best_num, true_labels)
    ari_scores['layout'] = layout_name

    return ari_scores



def full_cluster_experiment(G, true_labels):
    '''
    for ONE graph gets results of ALL layouts (all clustering algorithms)
    '''
    # df = pd.DataFrame(columns=['layout','AgglomerativeClustering', 'OPTICS', 'KMeans', 'GMM', 'Birch', 'Girvan Newman', 'Leiden'])
    df = pd.DataFrame(columns=['layout','AgglomerativeClustering', 'OPTICS', 'KMeans', 'GMM', 'Birch', 'Genie'])

    #for every layout
    #kamada kawai
    layout_names = ['kamada_kawai', 'spring', 'davidson_harel', 'drl', 'fruchterman_reingold', 'graphopt', 'lgl','mds']
    for layout_name in layout_names:
        ari_scores = calculate_scores_for_layout(G, true_labels, layout_name)
        df.loc[len(df)] = ari_scores
    return df

# generates k graphs and conducts FULL experiments on them
# it sums up ARIs and divides by k (average)
# returns : df

def steady_full_experiment(params, k=5, i_want_boxplot=False, dispersion=.35):
    '''
    generates k graphs and conducts FULL experiment on them
    it sums up ARIs and divides by k (average)
    returns

    layout	AgglomerativeClustering	OPTICS	KMeans	GMM	Birch	Girvan Newman	Leiden
0	kamada_kawai	0.436371	0.197825	0.518137	0.401987	0.276315	0.582956	0.155925
1	spring	0.385744	0.199351	0.382752	0.392318	0.155924	0.589295	0.146941
2	davidson_harel	0.167566	0.029073	0.179289	0.172008	0.168626	0.582956	0.146941
3	drl	0.065571	0.011393	0.071028	0.053928	0.031174	0.582956	0.146941
4	fruchterman_reingold	0.106105	0.037586	0.107198	0.080550	0.092003	0.582956	0.146941
5	graphopt	0.123411	0.053438	0.110400	0.106884	0.077662	0.582956	0.146941
6	lgl	0.021756	0.043951	0.021756	0.021756	0.019231	0.523917	0.155925
7	mds	0.111026	0.079521	0.102880	0.114435	0.130525	0.582956	0.146941

    '''
    # (n_vertex, n_comms, inside_prob, outside_prob) = params 
    (graph_id, n_vertex, n_comms, inside_prob, outside_prob) = params
    n_vertex = int(n_vertex)
    n_comms = int(n_comms)
    # (G, true_labels)= generate_G(sizes, inside_prob, outside_prob)

    #initiating first graph
    (G, true_labels) = generate_G_randomized(n_vertex, n_comms, inside_prob, outside_prob)
    asor = nx.numeric_assortativity_coefficient(G, "community")
    df = full_cluster_experiment(G, true_labels)
    df.iloc[:, 1:] = df.iloc[:, 1:].applymap(lambda x: [x])
    #comms - dict with two values
    # comms = get_communities(G, true_labels)
    comms = get_communities_scores_from_positions(G, true_labels)
    girvs = [comms['Girvan Newman']]
    leid = [comms['Leiden']]
    
    for i in tqdm(range(1, k)):
        (G, true_labels) = generate_G_randomized(n_vertex, n_comms, inside_prob, outside_prob)
        asor += nx.numeric_assortativity_coefficient(G, "community")
        tmp = full_cluster_experiment(G, true_labels)
        # df[df.columns[1:]] += tmp[tmp.columns[1:]]
        for column in df.columns[1:]:
            df[column] = df[column].combine(tmp[column], lambda x, y: x + [y])
        #@ separated community detection
        # tmp = get_communities(G, true_labels)
        tmp = get_communities_scores_from_positions(G, true_labels)
        girvs.append(tmp['Girvan Newman'])
        leid.append(tmp['Leiden'])
   
    # df[df.columns[1:]] /= k
    # df['Girvan-Newman'] = comms[0]/k
    df['Girvan-Newman'] = [girvs] * len(df)
    # df['Leiden'] = comms[1]/k
    df['Leiden'] = [leid] * len(df)
    
    if i_want_boxplot==False:
        # df.iloc[:, 1:] = df.iloc[:, 1:].applymap(lambda x: sum(x) / k)'
        df.iloc[:, 1:] = df.iloc[:, 1:].applymap(lambda x: sum(v for v in x if v != 'ERROR') / len([v for v in x if v != 'ERROR']))
    df['assortativity'] = asor/k

    return df
    
    
# def generate_G(sizes, inside_prob, outside_prob):
#     probs = np.eye(len(sizes)) * inside_prob

#     # Set the off-diagonal elements to the desired value (0.01)
#     probs[probs == 0] = outside_prob
#     true_labels=[]
#     i=0
#     for size in sizes:
#         true_labels += ([i]*size)
#         i += 1
#     G = nx.stochastic_block_model(sizes, probs, seed=random.randint(0, 200))
#     for node, community in zip(G.nodes(), true_labels):
#         G.nodes[node]['community'] = community
        
#     return (G, true_labels)

# if __name__== '__main__':
#     (G, true_labels) = generate_G_randomized(30, 3, .8, .02)
#     pos = nx.kamada_kawai_layout(G)
#     posdf = pd.DataFrame.from_dict(pos, orient='index', columns=['X', 'Y'])
#     best_num =3
#     # print(get_communities_scores_from_positions(G, true_labels))
#     print(full_cluster_experiment(G, true_labels))
    # print(f'Best number of clusters detected : {best_num}')
    # scores = get_clusters_from_positions(posdf, best_num, true_labels)