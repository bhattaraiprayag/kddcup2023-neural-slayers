# session_graph_builder.py

import math
import multiprocessing
import os
import pickle
from collections import Counter
from itertools import combinations
from typing import Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm

from configs import (
    GRAPH_FILENAME,
    GRAPH_TYPE,
    LOCALES,
    P2P_GRAPH_PATH,
    SLICER,
    TRAIN_PATH,
)

os.makedirs(P2P_GRAPH_PATH, exist_ok=True)


def _process_session_chunk(df_chunk: pd.DataFrame) -> tuple[Counter, Counter]:
    co_occurrence_counter = Counter()
    product_counts_counter = Counter()
    for _, row in df_chunk.iterrows():
        prev_items = (
            row["prev_items"].split(",") if isinstance(row["prev_items"], str) else []
        )
        all_items = prev_items + [row["next_item"]]
        unique_items = sorted(list(set(all_items)))
        product_counts_counter.update(unique_items)
        if len(unique_items) > 1:
            for pair in combinations(unique_items, 2):
                co_occurrence_counter[pair] += 1
    return co_occurrence_counter, product_counts_counter


class ProductGraphBuilder:
    def __init__(self, sessions_df: pd.DataFrame, products_df: pd.DataFrame):
        self.sessions_df = sessions_df.dropna(subset=["prev_items", "next_item"]).copy()
        self.products_df = products_df.set_index('id')
        self.products_df['attributes_text'] = (
            self.products_df['title'] + ' ' +
            self.products_df['locale'] + ' ' +
            self.products_df['price'].astype(str) + ' ' +
            self.products_df['brand'] + ' ' +
            self.products_df['color'] + ' ' +
            self.products_df['size'] + ' ' +
            self.products_df['model'] + ' ' +
            self.products_df['material'] + ' ' +
            self.products_df['desc']
        )
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.products_df['attributes_text'])
        self.product_to_tfidf_idx = {product_id: i for i, product_id in enumerate(self.products_df.index)}
        self.all_product_ids = set(products_df["id"].unique())

    def build_graph(
        self,
        weight_type: str = "co-occurrence",
        session_slice: Union[int, None] = None,
        num_workers: int = -1,
        alpha: float = 0.5
    ) -> nx.Graph:
        if weight_type not in ["co-occurrence", "pmi", "pmi-hybrid"]:
            raise ValueError("weight_type must be either 'co-occurrence' or 'pmi'")
        df = self.sessions_df
        if session_slice:
            print(
                f"Slicing dataframe to first {session_slice} sessions for prototyping."
            )
            df = df.head(session_slice)
        total_sessions = len(df)
        if total_sessions == 0:
            print("Warning: No sessions to process after cleaning and slicing.")
            G = nx.Graph()
            G.add_nodes_from(self.all_product_ids)
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
                    pool.imap(_process_session_chunk, df_chunks),
                    total=len(df_chunks),
                    desc="Processing chunks",
                )
            )
        total_co_occurrences = Counter()
        total_product_counts = Counter()
        for co_occurrence_counter, product_counts_counter in results:
            total_co_occurrences.update(co_occurrence_counter)
            total_product_counts.update(product_counts_counter)
        G = nx.Graph()
        G.add_nodes_from(self.all_product_ids)
        if weight_type == "co-occurrence":
            for (prod1, prod2), count in tqdm(
                total_co_occurrences.items(), desc="Adding Co-occurrence Edges"
            ):
                G.add_edge(prod1, prod2, weight=count)
        elif weight_type == "pmi":
            for (prod1, prod2), c_xy in tqdm(
                total_co_occurrences.items(), desc="Calculating PMI & Adding Edges"
            ):
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
        elif weight_type == "pmi-hybrid":
            print("Pre-calculating attribute similarities for all co-occurring pairs...")
            valid_pairs = [
                (p1, p2) for p1, p2 in total_co_occurrences.keys()
                if p1 in self.product_to_tfidf_idx and p2 in self.product_to_tfidf_idx
            ]
            if valid_pairs:
                indices1 = [self.product_to_tfidf_idx[p1] for p1, p2 in valid_pairs]
                indices2 = [self.product_to_tfidf_idx[p2] for p1, p2 in valid_pairs]
                tfidf_matrix1 = self.tfidf_matrix[indices1]
                tfidf_matrix2 = self.tfidf_matrix[indices2]
                sims_sparse = tfidf_matrix1.multiply(tfidf_matrix2).sum(axis=1)
                similarities = np.asarray(sims_sparse).flatten()
                attribute_similarity_lookup = dict(zip(valid_pairs, similarities))
            else:
                attribute_similarity_lookup = {}

            for (prod1, prod2), c_xy in tqdm(
                total_co_occurrences.items(), desc="Calculating Hybrid PMI & Adding Edges"
            ):
                c_x = total_product_counts[prod1]
                c_y = total_product_counts[prod2]
                if c_x > 0 and c_y > 0:
                    p_xy = c_xy / total_sessions
                    p_x = c_x / total_sessions
                    p_y = c_y / total_sessions
                    if p_xy > 0:
                        pmi = math.log2(p_xy / (p_x * p_y))
                        ppmi = max(0, pmi)
                    else:
                        ppmi = 0
                    attribute_similarity = attribute_similarity_lookup.get((prod1, prod2), 0)
                    final_weight = (alpha * ppmi) + ((1 - alpha) * attribute_similarity)
                    if final_weight > 0:
                        G.add_edge(prod1, prod2, weight=final_weight)
        return G

    @staticmethod
    def save_graph(graph: nx.Graph, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(graph, f)

    @staticmethod
    def load_graph(filepath: str) -> nx.Graph:
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
        if os.path.exists(graph_file_path):
            print(f"Graph for {locale} already exists at {graph_file_path}. Skipping.")
            continue
        print(f"\nLocale: {locale}")
        locale_sessions_df = sessions_df[sessions_df["locale"] == locale].copy()
        locale_products_df = products_df[products_df["locale"] == locale]

        ### DEBUG ###
        all_catalog_products = set(locale_products_df["id"].unique())
        prev_items_set = set(
            locale_sessions_df["prev_items"].str.split(",").explode().unique()
        )
        next_items_set = set(locale_sessions_df["next_item"].unique())
        all_session_items = prev_items_set.union(next_items_set) - {pd.NA, None, ""}
        missing_products = all_catalog_products - all_session_items
        if missing_products:
            print(
                f"{len(missing_products)} products in the products catalog for {locale} are not present in its sessions."
            )
        ### DEBUG ###

        graph_builder = ProductGraphBuilder(locale_sessions_df, locale_products_df)
        p2p_graph = graph_builder.build_graph(
            weight_type=GRAPH_TYPE,
            num_workers=-1,
            # session_slice=SLICER
        )
        # print(f"\nGraph info ({GRAPH_TYPE.upper()}):")
        # pair_to_check = sample_values.get(locale, [])
        # print(f"Edge weights for {pair_to_check[0]} and {pair_to_check[1]}:",
        #       p2p_graph.get_edge_data(pair_to_check[0], pair_to_check[1]))
        # print(f"Edge weights for {pair_to_check[0]} and {pair_to_check[2]}:",
        #         p2p_graph.get_edge_data(pair_to_check[0], pair_to_check[2]))
        ProductGraphBuilder.save_graph(p2p_graph, graph_file_path)
        print(f"===" * 35)
