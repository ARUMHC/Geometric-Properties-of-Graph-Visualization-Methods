import os
import sys
sys.path.insert(0, r'C:\Users\Kinga\Desktop\MAGISTERKA\Geometric-Properties-of-Graph-Visualization-Methods\code')
import pandas as pd
from clustering_script import *

import warnings
import multiprocessing as mp
from tqdm import tqdm

warnings.filterwarnings('ignore')

def process_params(params):
    _, params = params
    params = tuple(params)
    single_df = steady_full_experiment(params, k=2, i_want_boxplot=False, dispersion=.35)
    single_df['graph_id'] = int(params[0])
    return single_df

def main(graph_params, size):
    all_graphs_df = pd.DataFrame()

    # Use multiprocessing to process each set of parameters
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_params, graph_params.iterrows()), total=len(graph_params)))

    # Concatenate all results into a single DataFrame
    for single_df in results:
        print(f'GRAPH ID : {single_df["graph_id"][0]}')
        all_graphs_df = pd.concat([all_graphs_df, single_df], axis=0)

    # Save the final DataFrame to an Excel file
    all_graphs_df.to_excel(f'whole_experiment/{size}_steady_ex.xlsx', index=False)

# if __name__ == '__main__':
#     main(graph_params)