from graph_generating_script import *
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')

from yellowbrick.cluster import KElbowVisualizer
from sklearn.mixture import GaussianMixture

'''
Five different algorithms to choose best number of clusters:
* gap statistic
* silhouette
* elbow method
* calinski harabasz

all those functions take posdf and max_clusters as arguments
all return optimal numbers of clusters and dataframe with scores
'''

def gap_statistic_best_num(posdf, max_clusters=10, nrefs=3):
        gaps = np.zeros((len(range(1, max_clusters)),))
        df_scores = pd.DataFrame({'clusterCount':[], 'gap':[]})
        for gap_index, k in enumerate(range(1, max_clusters)):
                # Holder for reference dispersion results
                refDisps = np.zeros(nrefs)
                # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
                for i in range(nrefs):
                        # Create new random reference set
                        randomReference = np.random.random_sample(size=posdf.shape)
                        # Fit to it
                        km = KMeans(k)
                        km.fit(randomReference)
                        
                        refDisp = km.inertia_
                        refDisps[i] = refDisp
                        # Fit cluster to original data and create dispersion
                km = KMeans(k)
                km.fit(posdf)
                
                origDisp = km.inertia_
                # Calculate gap statistic
                gap = np.log(np.mean(refDisps)) - np.log(origDisp)
                # Assign this loop's gap statistic to gaps
                gaps[gap_index] = gap
                
                new_row = pd.DataFrame({'clusterCount': [k], 'gap': [gap]})
                df_scores = pd.concat([df_scores, new_row], ignore_index=True)  
        
        return (gaps.argmax() + 1, df_scores)


def elbow_method_best_num(posdf, max_clusters=10):
    model = KMeans()
    # k is range of number of clusters.
    visualizer = KElbowVisualizer(model, k=(2, max_clusters), timings= True)
    # the plot kept showing under the results
    plt.ioff()
    visualizer = visualizer.fit(posdf)
    plt.ion()
    plt.gcf().clear()       
    # visualizer.show()    
    scores=visualizer.k_scores_
    df_scores = pd.DataFrame({'k': range(2, max_clusters), 'score': scores})
    if visualizer.elbow_value_ is None:
        # Return the k with the minimum score (inertia)
        best_k = df_scores.loc[df_scores['score'].idxmin()]['k']
    else:
        best_k = int(visualizer.elbow_value_)
    
    return(int(best_k), df_scores)

def silhouette_best_num(posdf, max_clusters=10):
    model = KMeans()
    # k is range of number of clusters.
    visualizer = KElbowVisualizer(model, k=(2, max_clusters),metric='silhouette', timings= True)
    plt.ioff()
    visualizer.fit(posdf)
    plt.ion()
    plt.gcf().clear()        
    # visualizer.show() 
    scores = visualizer.k_scores_
    df_scores = pd.DataFrame({'k': range(2, max_clusters), 'score': scores})
    if visualizer.elbow_value_ is None:
         best_row = df_scores.loc[df_scores['score'].idxmax()]
         best_num = best_row['k']
    else:
         best_num = visualizer.elbow_value_
    return(int(best_num), df_scores)


def calinski_harabasz_best_num(posdf, max_clusters=10):
    model = KMeans()
    # k is range of number of clusters.
    visualizer = KElbowVisualizer(model, k=(2, max_clusters), metric='calinski_harabasz', timings= True)
    plt.ioff()
    visualizer.fit(posdf)        
    plt.ion()
    plt.gcf().clear()        
    # visualizer.show() 
    scores = visualizer.k_scores_
    df_scores = pd.DataFrame({'k': range(2, max_clusters), 'score': scores})
    if visualizer.elbow_value_ is None:
        # Return the k with the highest score (Calinski-Harabasz index)
        best_k = df_scores.loc[df_scores['score'].idxmax()]['k']
    else:
        best_k = int(visualizer.elbow_value_)
    
    return(best_k, df_scores)   


def mix_ch_elbow(posdf, coef_ch = .5, coef_elbow = .5, max_clusters=10):
     best_k_ci , _ = calinski_harabasz_best_num(posdf, max_clusters)
     best_k_elbow, _  = elbow_method_best_num(posdf, max_clusters)
     return round(coef_ch*best_k_ci + coef_elbow*best_k_elbow)



def BIC_best_num(posdf, max_clusters=10):
    n_components = range(2, max_clusters)
    covariance_type = ['spherical', 'tied', 'diag', 'full']
    score = []

    for cov in covariance_type:
        for n_comp in n_components:
            gmm = GaussianMixture(n_components=n_comp, covariance_type=cov)
            gmm.fit(posdf)
            score.append((cov, n_comp, gmm.bic(posdf)))

    # Convert the score list to a DataFrame
    df_scores = pd.DataFrame(score, columns=['covariance_type', 'n_components', 'bic'])
    min_bic_row = df_scores.loc[df_scores['bic'].idxmin()]

    # Extract the optimal covariance type and number of components
    # optimal_covariance_type = min_bic_row['covariance_type']
    optimal_n_components = min_bic_row['n_components']
    # optimal_bic = min_bic_row['bic']

    return (int(optimal_n_components), score)

from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans
