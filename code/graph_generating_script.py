import random
import numpy as np   
import networkx as nx

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