import sys
sys.path.insert(0, r'C:\Users\Kinga\Desktop\MAGISTERKA\Geometric-Properties-of-Graph-Visualization-Methods\code')
from clustering_script import *
from graph_generating_script import *

from clus_num_methods import *

from tqdm import tqdm 

class BestNumExperiment():
    def __init__(self, graph_params:pd.DataFrame):
        self.graph_params = graph_params
        self.graph_posdfs = {}

    def calculate_posdfs(self, from_file=None):
        if from_file==None:
            for _, row in tqdm(self.graph_params.iterrows(), total=self.graph_params.shape[0]):
                print(row['graph_id'])
                (G, true_labels) = generate_G_randomized(int(row['size']), int(row['no_comms']), row['inside_prob'], row['outside_prob'])
                self.graph_posdfs[row['graph_id']] = {}
                assor = nx.numeric_assortativity_coefficient(G, "community")
                self.graph_posdfs[row['graph_id']]['assortativity'] = assor
                layout_names = ['kamada_kawai', 'spring', 'davidson_harel', 'drl', 'fruchterman_reingold', 'graphopt', 'lgl','mds']
                for layout_name in layout_names:
                    posdf = posdf_from_layout(G, layout_name)
                    
                    # Store the posdf DataFrame in the nested dictionary
                    self.graph_posdfs[row['graph_id']][layout_name] = posdf
        else:
            self.graph_posdfs
    

    def make_experiment(self, best_num_algo_name):
        #iterating through all the graphs
        results = pd.DataFrame(columns=['graph_id', 'assortativity', 'layout_name', 'no_communities', 'calculated_bestnum'])
        for _, row in tqdm(self.graph_params.iterrows(), total=self.graph_params.shape[0]):
        # for _, row in self.graph_params.iterrows():
            # (G, true_labels) = generate_G_randomized(int(row['size']), int(row['no_comms']), row['inside_prob'], row['outside_prob'])
            # assor = nx.numeric_assortativity_coefficient(G, "community")
            layout_names = ['kamada_kawai', 'spring', 'davidson_harel', 'drl', 'fruchterman_reingold', 'graphopt', 'lgl','mds']
            for layout_name in layout_names:
                # posdf = posdf_from_layout(G, layout_name)
                posdf = self.graph_posdfs[row['graph_id']][layout_name]
                if best_num_algo_name == 'gap_statistic':
                    (best_num, _) = gap_statistic_best_num(posdf)
                elif best_num_algo_name == 'elbow_method':
                    (best_num, _) = elbow_method_best_num(posdf)
                elif best_num_algo_name == 'silhouette':
                    (best_num, _) = silhouette_best_num(posdf)
                elif best_num_algo_name == 'calinski_harabasz':
                    (best_num, _) = calinski_harabasz_best_num(posdf)
                elif best_num_algo_name == '50_mix_ch_elbow':
                    best_num = mix_ch_elbow(posdf, .5, .5)
                elif best_num_algo_name == '75_mix_ch_elbow':
                    best_num = mix_ch_elbow(posdf, .75, .25)
                # elif best_num_algo_name == 'BIC':
                    # (best_num, _) = BIC_best_num(posdf)
                else:
                    raise ValueError('Incorrect algorith name, probably a typo')

                new_row = {'graph_id':[int(row['graph_id'])], 
                           'assortativity': [self.graph_posdfs[row['graph_id']]['assortativity']], 
                           'size' : [int(row['size'])],
                           'layout_name': [layout_name], 
                           'no_communities': [int(row['no_comms'])],
                           'calculated_bestnum':[int(best_num)]}
                # print(new_row)
                results = pd.concat([results, pd.DataFrame(new_row)])
                
        if best_num_algo_name == 'gap_statistic':
            self.gap= results
        elif best_num_algo_name == 'elbow_method':
            self.elbow = results
        elif best_num_algo_name == 'silhouette':
            self.silhouette = results
        elif best_num_algo_name == 'calinski_harabasz':
            self.ch = results
        elif best_num_algo_name == '50_mix_ch_elbow':
            self.mix_ch_elbow_50 = results
        elif best_num_algo_name == '75_mix_ch_elbow':
            self.mix_ch_elbow_75 = results
        # elif best_num_algo_name == 'BIC':
            # self.bic = results

import warnings
warnings.filterwarnings('ignore')

def calculate_results_for_size(graph_params, size):

    print(f'======= GRAPHS SIZE {size} ==========')
    ex1 = BestNumExperiment(graph_params)
    ex1.calculate_posdfs()

    ex1.make_experiment('gap_statistic')
    ex1.gap.to_csv(fr'code\NUM_CLUST_choosing_method\clus_num_data\{size}_gap_results.csv', index=False)
    ex1.make_experiment('elbow_method')
    ex1.elbow.to_csv(fr'code\NUM_CLUST_choosing_method\clus_num_data\{size}_elbow_results.csv', index=False)
    ex1.make_experiment('silhouette')
    ex1.silhouette.to_csv(fr'code\NUM_CLUST_choosing_method\clus_num_data\{size}_silhouette_results.csv', index=False)
    ex1.make_experiment('calinski_harabasz')
    ex1.ch.to_csv(fr'code\NUM_CLUST_choosing_method\clus_num_data\{size}_ch_results.csv', index=False)
    ex1.make_experiment('50_mix_ch_elbow')
    ex1.mix_ch_elbow_50.to_csv(fr'code\NUM_CLUST_choosing_method\clus_num_data\{size}_50_mix_ch_elbow_results.csv', index=False)
    ex1.make_experiment('75_mix_ch_elbow')
    ex1.mix_ch_elbow_75.to_csv(fr'code\NUM_CLUST_choosing_method\clus_num_data\{size}_75_mix_ch_elbow_results.csv', index=False)


import os
print(os.getcwd())
graph_params = pd.read_excel(r'code\params\25_graph_params.xlsx')
calculate_results_for_size(graph_params, '25')

graph_params = pd.read_excel(r'code\params\50_graph_params.xlsx')
calculate_results_for_size(graph_params, '50')

graph_params = pd.read_excel(r'code\params\75_graph_params.xlsx')
calculate_results_for_size(graph_params, '75')

graph_params = pd.read_excel(r'code\params\100_graph_params.xlsx')
calculate_results_for_size(graph_params, '100')

graph_params = pd.read_excel(r'code\params\125_graph_params.xlsx')
calculate_results_for_size(graph_params, '125')

graph_params = pd.read_excel(r'code\params\150_graph_params.xlsx')
calculate_results_for_size(graph_params, '150')


# ex1 = BestNumExperiment(graph_params)
# ex1.calculate_posdfs()

# size='50'
# ex1.make_experiment('gap_statistic')
# ex1.gap.to_csv(fr'data\{size}_gap_results.csv', index=False)
# ex1.make_experiment('elbow_method')
# ex1.elbow.to_csv(fr'data\{size}_elbow_results.csv', index=False)
# ex1.make_experiment('silhouette')
# ex1.silhouette.to_csv(fr'data\{size}_silhouette_results.csv', index=False)
# ex1.make_experiment('calinski_harabasz')
# ex1.ch.to_csv(fr'data\{size}_ch_results.csv', index=False)
# ex1.make_experiment('50_mix_ch_elbow')
# ex1.mix_ch_elbow_50.to_csv(fr'data\{size}_50_mix_ch_elbow_results.csv', index=False)
# ex1.make_experiment('75_mix_ch_elbow')
# ex1.mix_ch_elbow_75.to_csv(fr'data\{size}_75_mix_ch_elbow_results.csv', index=False)


# graph_params = pd.read_excel(r'params\80_graph_params.xlsx')

# ex1 = BestNumExperiment(graph_params)
# ex1.calculate_posdfs()

# size='80'
# ex1.make_experiment('gap_statistic')
# ex1.gap.to_csv(fr'data\{size}_gap_results.csv', index=False)
# ex1.make_experiment('elbow_method')
# ex1.elbow.to_csv(fr'data\{size}_elbow_results.csv', index=False)
# ex1.make_experiment('silhouette')
# ex1.silhouette.to_csv(fr'data\{size}_silhouette_results.csv', index=False)
# ex1.make_experiment('calinski_harabasz')
# ex1.ch.to_csv(fr'data\{size}_ch_results.csv', index=False)
# ex1.make_experiment('50_mix_ch_elbow')
# ex1.mix_ch_elbow_50.to_csv(fr'data\{size}_50_mix_ch_elbow_results.csv', index=False)
# ex1.make_experiment('75_mix_ch_elbow')
# ex1.mix_ch_elbow_75.to_csv(fr'data\{size}_75_mix_ch_elbow_results.csv', index=False)

# graph_params = pd.read_excel(r'params\100_graph_params.xlsx')

# ex1 = BestNumExperiment(graph_params)
# ex1.calculate_posdfs()

# size='100'
# ex1.make_experiment('gap_statistic')
# ex1.gap.to_csv(fr'data\{size}_gap_results.csv', index=False)
# ex1.make_experiment('elbow_method')
# ex1.elbow.to_csv(fr'data\{size}_elbow_results.csv', index=False)
# ex1.make_experiment('silhouette')
# ex1.silhouette.to_csv(fr'data\{size}_silhouette_results.csv', index=False)
# ex1.make_experiment('calinski_harabasz')
# ex1.ch.to_csv(fr'data\{size}_ch_results.csv', index=False)
# ex1.make_experiment('50_mix_ch_elbow')
# ex1.mix_ch_elbow_50.to_csv(fr'data\{size}_50_mix_ch_elbow_results.csv', index=False)
# ex1.make_experiment('75_mix_ch_elbow')
# ex1.mix_ch_elbow_75.to_csv(fr'data\{size}_75_mix_ch_elbow_results.csv', index=False)


# graph_params = pd.read_excel(r'params\150_graph_params.xlsx')

# ex1 = BestNumExperiment(graph_params)
# ex1.calculate_posdfs()

# size='150'
# ex1.make_experiment('gap_statistic')
# ex1.gap.to_csv(fr'data\{size}_gap_results.csv', index=False)
# ex1.make_experiment('elbow_method')
# ex1.elbow.to_csv(fr'data\{size}_elbow_results.csv', index=False)
# ex1.make_experiment('silhouette')
# ex1.silhouette.to_csv(fr'data\{size}_silhouette_results.csv', index=False)
# ex1.make_experiment('calinski_harabasz')
# ex1.ch.to_csv(fr'data\{size}_ch_results.csv', index=False)
# ex1.make_experiment('50_mix_ch_elbow')
# ex1.mix_ch_elbow_50.to_csv(fr'data\{size}_50_mix_ch_elbow_results.csv', index=False)
# ex1.make_experiment('75_mix_ch_elbow')
# ex1.mix_ch_elbow_75.to_csv(fr'data\{size}_75_mix_ch_elbow_results.csv', index=False)