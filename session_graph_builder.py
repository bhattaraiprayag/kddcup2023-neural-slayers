# session_graph_builder.py

import math
import multiprocessing
import os
import pickle
from collections import Counter
from itertools import combinations
from typing import Union

import networkx as nx
import pandas as pd
from tqdm import tqdm

from configs import LOCALES, P2P_GRAPH_PATH, SLICER, TRAIN_PATH

os.makedirs(P2P_GRAPH_PATH, exist_ok=True)

GRAPH_TYPE = "pmi"  # ["co-occurrence", "pmi"]
GRAPH_FILENAME = f"graph_{GRAPH_TYPE}.gpickle"


def _process_session_chunk(df_chunk: pd.DataFrame) -> tuple[Counter, Counter]:
    co_occurrence_counter = Counter()
    product_counts_counter = Counter()
    for _, row in df_chunk.iterrows():
        prev_items = row['prev_items'].split(',') if isinstance(row['prev_items'], str) else []
        all_items = prev_items + [row['next_item']]
        unique_items = sorted(list(set(all_items)))
        product_counts_counter.update(unique_items)
        if len(unique_items) > 1:
            for pair in combinations(unique_items, 2):
                co_occurrence_counter[pair] += 1
    return co_occurrence_counter, product_counts_counter


class ProductGraphBuilder:
    """
    Builds a weighted, undirected product-product graph from session data.
    """
    def __init__(self, sessions_df: pd.DataFrame):
        self.sessions_df = sessions_df.dropna(subset=['prev_items', 'next_item']).copy()

    def build_graph(self, 
                    weight_type: str = 'co-occurrence', 
                    session_slice: Union[int, None] = None, 
                    num_workers: int = -1) -> nx.Graph:
        if weight_type not in ['co-occurrence', 'pmi']:
            raise ValueError("weight_type must be either 'co-occurrence' or 'pmi'")

        df = self.sessions_df
        if session_slice:
            print(f"Slicing dataframe to first {session_slice} sessions for prototyping.")
            df = df.head(session_slice)

        total_sessions = len(df)
        if total_sessions == 0:
            print("Warning: No sessions to process after cleaning and slicing.")
            return nx.Graph()

        if num_workers == -1:
            num_workers = multiprocessing.cpu_count()
        print(f"Using {num_workers} worker processes.")

        chunk_size = max(1, total_sessions // num_workers)
        df_chunks = [df.iloc[i:i + chunk_size] for i in range(0, total_sessions, chunk_size)]

        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(_process_session_chunk, df_chunks), total=len(df_chunks), desc="Processing chunks"))

        total_co_occurrences = Counter()
        total_product_counts = Counter()
        for co_occurrence_counter, product_counts_counter in results:
            total_co_occurrences.update(co_occurrence_counter)
            total_product_counts.update(product_counts_counter)     
        print(f"Found: {len(total_co_occurrences)} unique co-occurring pairs | {len(total_product_counts)} unique products.")

        G = nx.Graph()        
        G.add_nodes_from(total_product_counts.keys())

        if weight_type == 'co-occurrence':
            for (prod1, prod2), count in tqdm(total_co_occurrences.items(), desc="Adding Co-occurrence Edges"):
                G.add_edge(prod1, prod2, weight=count)
        elif weight_type == 'pmi':
            for (prod1, prod2), c_xy in tqdm(total_co_occurrences.items(), desc="Calculating PMI & Adding Edges"):
                c_x = total_product_counts[prod1]
                c_y = total_product_counts[prod2]
                if c_x > 0 and c_y > 0:
                    p_xy = c_xy / total_sessions
                    p_x = c_x / total_sessions
                    p_y = c_y / total_sessions
                    pmi = math.log2(p_xy / (p_x * p_y))
                    ppmi = max(0, pmi)
                    if ppmi > 0:
                        G.add_edge(prod1, prod2, weight=ppmi)
        return G

    @staticmethod
    def save_graph(graph: nx.Graph, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(graph, f)

    @staticmethod
    def load_graph(filepath: str) -> nx.Graph:
        with open(filepath, 'rb') as f:
            graph = pickle.load(f)
        return graph


if __name__ == '__main__':
    sessions_data_path = TRAIN_PATH + 'sessions_train_processed.csv'
    sessions_df = pd.read_csv(sessions_data_path)

    sample_values = {
        "DE": ['B09W9FND7K', 'B09JSPLN1M', 'B09M7GY217'],
        "JP": ['B08CK9HRV6', 'B09ZD6YJJP', 'B0B87D412M'],
        "UK": ['B0BFDL54Y7', 'B0BFDR9X13', 'B07J4WF8VH']
    }
    sessions_df = sessions_df[sessions_df['locale'].isin(LOCALES)]

    for locale in LOCALES:
        print(f"\nProcessing locale: {locale} | Initial sessions: {len(sessions_df)}")
        locale_sessions_df = sessions_df[sessions_df['locale'] == locale].copy()

        graph_builder = ProductGraphBuilder(locale_sessions_df)
        p2p_graph = graph_builder.build_graph(
            weight_type=GRAPH_TYPE,
            num_workers=-1,
            # session_slice=SLICER
        )

        print(f"\nGraph info ({GRAPH_TYPE.upper()}):")
        pair_to_check = sample_values.get(locale, [])
        print(f"Edge weights for {pair_to_check[0]} and {pair_to_check[1]}:", 
              p2p_graph.get_edge_data(pair_to_check[0], pair_to_check[1]))
        print(f"Edge weights for {pair_to_check[0]} and {pair_to_check[2]}:",
                p2p_graph.get_edge_data(pair_to_check[0], pair_to_check[2]))
        graph_file_path = P2P_GRAPH_PATH + GRAPH_FILENAME.replace('.gpickle', f'_{locale}.gpickle')
        ProductGraphBuilder.save_graph(p2p_graph, graph_file_path)
        print(f"===" * 35)

        # loaded_co_graph = ProductGraphBuilder.load_graph(COOCCURRENCE_GRAPH_FILE_PATH)
        # loaded_pmi_graph = ProductGraphBuilder.load_graph(PMI_GRAPH_FILE_PATH)
