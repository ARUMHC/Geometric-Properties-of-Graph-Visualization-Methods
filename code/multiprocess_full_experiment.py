import os
import sys
from pathlib import Path
# sys.path.insert(0, r'C:\Users\ku51015\CHMURA\mystuff\geaometricgraphproperties\code')
sys.path.insert(0, str(Path(r'C:\Users\Kinga\Desktop\MAGISTERKA\Geometric-Properties-of-Graph-Visualization-Methods\code').resolve()))
import time
import pandas as pd
from clustering_script import *

import warnings
import multiprocessing as mp
from tqdm import tqdm

warnings.filterwarnings('ignore')

def process_params(params):
    _, params = params
    #@ PLACE THAT SETS THE VALUE OF k
    k = 5
    params = tuple(params)
    single_df = steady_full_experiment(params, k=k, i_want_boxplot=False, dispersion=.35)
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
    all_graphs_df.to_csv(fr'C:\Users\Kinga\Desktop\MAGISTERKA\Geometric-Properties-of-Graph-Visualization-Methods\code\whole_experiment\{size}_steady_ex.csv', index=False)
if __name__ == '__main__':

    #@ PLACE THAT SETS THE SIZES TO RUN
    sizes_to_run = [25, 50, 75, 100, 125, 150]

    for size_to_run in sizes_to_run:
        print(f"=== processing size {size_to_run} ===")
        start_time = time.time()  # Record the start time
        # graph_params = pd.read_excel(rf'params/{size_to_run}_graph_params.xlsx')
        graph_params = pd.read_excel(fr'C:\Users\Kinga\Desktop\MAGISTERKA\Geometric-Properties-of-Graph-Visualization-Methods\code\params\{size_to_run}_graph_params.xlsx')
        main(graph_params, size=size_to_run)
        end_time = time.time()  
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time,60)
        print(f"Total execution time: {int(minutes)} minutes and {seconds:.2f} seconds")
        # print(f"It took {elapsed_time} seconds to process size {size_to_run}")