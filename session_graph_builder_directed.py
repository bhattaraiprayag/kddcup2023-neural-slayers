# session_graph_builder_directed.py

import math
import multiprocessing
import os
import pickle
from collections import Counter
from typing import Union

import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm

from configs import (
    GRAPH_FILENAME,
    LOCALES,
    P2P_GRAPH_PATH,
    SLICER,
    TRAIN_PATH,
)

os.makedirs(P2P_GRAPH_PATH, exist_ok=True)


def _process_session_chunk_directed(
    df_chunk: pd.DataFrame,
) -> tuple[Counter, Counter, Counter, Counter]:
    directed_co_occurrence_counter = Counter()
    product_session_counter = Counter()
    source_product_counter = Counter()
    target_product_counter = Counter()
    for _, row in df_chunk.iterrows():
        prev_items = (
            row["prev_items"].split(",") if isinstance(row["prev_items"], str) else []
        )
        all_items = prev_items + [row["next_item"]]
        unique_items = list(set(all_items))
        product_session_counter.update(unique_items)
        if len(all_items) > 1:
            for i in range(len(all_items) - 1):
                source_item = all_items[i]
                target_item = all_items[i + 1]
                if source_item != target_item:
                    pair = (source_item, target_item)
                    directed_co_occurrence_counter[pair] += 1
                    source_product_counter[source_item] += 1
                    target_product_counter[target_item] += 1
    return (
        directed_co_occurrence_counter, product_session_counter,
        source_product_counter, target_product_counter,
    )


class ProductGraphBuilder:
    def __init__(self, sessions_df: pd.DataFrame, products_df: pd.DataFrame):
        self.sessions_df = sessions_df.dropna(subset=["prev_items", "next_item"]).copy()
        self.products_df = products_df.set_index('id')
        self.all_product_ids = set(self.products_df.index.unique())

    def build_graph(
        self,
        session_slice: Union[int, None] = None,
        num_workers: int = -1,
    ) -> nx.DiGraph:
        df = self.sessions_df
        if session_slice:
            print(f"Slicing dataframe to first {session_slice} sessions for prototyping.")
            df = df.head(session_slice)

        total_sessions = len(df)
        G = nx.DiGraph()
        for product_id, attributes in tqdm(
            self.products_df.to_dict('index').items(), desc="Adding Product Nodes"
        ):
            G.add_node(product_id, **attributes)
        if total_sessions == 0:
            print("Warning: No sessions to process. Returning a graph with nodes only.")
            return G
        if num_workers == -1:
            num_workers = multiprocessing.cpu_count()
        chunk_size = max(1, total_sessions // num_workers)
        df_chunks = [
            df.iloc[i : i + chunk_size] for i in range(0, total_sessions, chunk_size)
        ]

        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_process_session_chunk_directed, df_chunks),
                    total=len(df_chunks),
                    desc="Processing chunks",
                )
            )
        total_directed_co_occurrences = Counter()
        total_product_counts = Counter()
        total_source_counts = Counter()
        total_target_counts = Counter()
        for (
            co_occurrence_counter,
            product_counts_counter,
            source_counter,
            target_counter,
        ) in results:
            total_directed_co_occurrences.update(co_occurrence_counter)
            total_product_counts.update(product_counts_counter)
            total_source_counts.update(source_counter)
            total_target_counts.update(target_counter)

        total_pairs = sum(total_directed_co_occurrences.values())
        if total_pairs == 0:
            print("Warning: No directed pairs found. Returning graph with nodes only.")
            return G
        for (source, target), count in tqdm(
            total_directed_co_occurrences.items(), desc="Calculating PPMI & Adding Edges"
        ):
            c_st = count
            c_s = total_source_counts[source]
            c_t = total_target_counts[target]
            
            if c_s > 0 and c_t > 0:
                p_st = c_st / total_pairs
                p_s = c_s / total_pairs
                p_t = c_t / total_pairs
                pmi = math.log2(p_st / (p_s * p_t))
                ppmi = max(0, pmi)
                if ppmi > 0:
                    G.add_edge(
                        source,
                        target,
                        cooccurrence_count=c_st,
                        ppmi_weight=ppmi
                    )
        return G

    @staticmethod
    def save_graph(graph: nx.DiGraph, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(graph, f)

    @staticmethod
    def load_graph(filepath: str) -> nx.DiGraph:
        with open(filepath, "rb") as f:
            graph = pickle.load(f)
        return graph


if __name__ == "__main__":
    sessions_data_path = TRAIN_PATH + "sessions_train_processed.csv"
    products_data_path = TRAIN_PATH + "products_train_processed.csv"
    
    sessions_df = pd.read_csv(sessions_data_path)
    products_df = pd.read_csv(products_data_path)
    sample_values = {
        "DE": ["B09W9FND7K", "B09JSPLN1M", "B09M7GY217"],
        "JP": ["B08CK9HRV6", "B09ZD6YJJP", "B0B87D412M"],
        "UK": ["B0BFDL54Y7", "B0BFDR9X13", "B07J4WF8VH"],
    }

    sessions_df = sessions_df[sessions_df["locale"].isin(LOCALES)]
    products_df = products_df[products_df["locale"].isin(LOCALES)]

    for locale in LOCALES:
        graph_file_path = P2P_GRAPH_PATH + GRAPH_FILENAME.replace(
            ".gpickle", f"_{locale}.gpickle"
        )
        print(f"\nProcessing Locale: {locale.upper()}")
        locale_sessions_df = sessions_df[sessions_df["locale"] == locale].copy()
        locale_products_df = products_df[products_df["locale"] == locale]

        graph_builder = ProductGraphBuilder(locale_sessions_df, locale_products_df)
        if os.path.exists(graph_file_path):
            print(f"Graph for {locale} already exists. Loading...")
            p2p_graph = graph_builder.load_graph(graph_file_path)
        else:
            print(f"Building graph for {locale}...")
            p2p_graph = graph_builder.build_graph(
                num_workers=-1,
                # session_slice=SLICER # Uncomment for quick testing
            )
            ProductGraphBuilder.save_graph(p2p_graph, graph_file_path)

        print(f"\nSample Graph info for ({locale.upper()}):")
        print(f"Number of nodes: {p2p_graph.number_of_nodes()}")
        print(f"Number of edges: {p2p_graph.number_of_edges()}")
        pair_to_check = sample_values.get(locale, [])
        print(f"Edge weights for {pair_to_check[0]} and {pair_to_check[1]}:",
              p2p_graph.get_edge_data(pair_to_check[0], pair_to_check[1]))
        print(f"Edge weights for {pair_to_check[0]} and {pair_to_check[2]}:",
                p2p_graph.get_edge_data(pair_to_check[0], pair_to_check[2]))

        print("\n--- Specific Node Information ---")
        for node_id in pair_to_check:
            if node_id in p2p_graph:
                print(f"Node: {node_id}, Data: {p2p_graph.nodes[node_id]}")
            else:
                print(f"Node: {node_id} not found in the graph.")
        print(f"===" * 30)
