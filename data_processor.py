# data_processor.py

import os
import re
from typing import List, Tuple

import pandas as pd


def handle_data(
    project_files: List[str],
    file_paths: List[str],
    task: str,
    task1_locales: List[str],
    fraction: float,
    seed: int,
    prod_dtypes: dict,
    sess_dtypes: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    def check_processed(file: str, path: str) -> bool:
        processed_file = os.path.join(path, file.split('.')[0] + '_processed.csv')
        return os.path.exists(processed_file)

    def get_processed_file_name(file: str) -> str:
        return file.split('.')[0] + '_processed.csv'

    def read_file(
        file: str,
        path: str,
        task: str,
        task1_locales: List[str],
        fraction: float,
        seed: int,
        dtypes: dict
    ) -> pd.DataFrame:
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path, dtype=dtypes)
        df = df[df['locale'].isin(task1_locales)]
        if fraction < 1:
            df = df.sample(frac=fraction, random_state=seed)
        return df

    def process_data(file: str, path: str, task: str, dtypes: dict) -> None:
        file_path = os.path.join(path, file)
        processed_file = os.path.join(path, file.split('.')[0] + '_processed.csv')
        df = pd.read_csv(file_path, dtype=dtypes)
        if 'products_train.csv' in file:
            df.replace(["Nan", "nan", "Null", "N/a", "NaN", "n/A"], pd.NA, inplace=True)
            df.fillna('-', inplace=True)
            non_text_cols = ['id', 'locale', 'price']
            text_cols = [col for col in df.columns if col not in non_text_cols]
            def clean_string(text: str) -> str:
                english = r'[a-zA-Z0-9\u0020]'
                german = r'[\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u00FF\u1E00-\u1EFF\u0020]'
                japanese = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u0020]'
                french = r'[a-zA-Z\u00C0-\u00FF\u0020]'
                spanish = r'[a-zA-Z\u00C0-\u00FF\u0020]'
                allowed_chars = english + german + japanese + french + spanish
                cleaned_text = re.sub(f"[^{allowed_chars}]+", " ", text)
                return cleaned_text
            for column in text_cols:
                if column in text_cols:
                    df[column] = df[column].apply(lambda x: clean_string(x) if x != '-' else x)
                    df[column] = df[column].str.lower()
        elif 'sessions_train.csv' in file or f'sessions_test_{task}.csv' in file:
            df['session_id'] = df.index + 1
            df['session_id'] = df['session_id'].astype('int32')
            df = df[[df.columns[-1]] + list(df.columns[:-1])]
            df['prev_items'] = df['prev_items'].str.findall(r"'(\w+)'").apply(lambda x: ','.join(x))
            df = df[['session_id', 'locale', 'prev_items', 'next_item']] if 'sessions_train.csv' in file else df[['session_id', 'locale', 'prev_items']]
        try:
            df.to_csv(processed_file, index=False)
        except Exception as e:
            print(f'Error saving processed file: {processed_file.split("/")[-1]} to: {path} due to {e}')

    products_file, sessions_file, sessions_test_file = project_files
    products_path, sessions_path, sessions_test_path = file_paths

    if check_processed(products_file, products_path):
        products = read_file(get_processed_file_name(products_file), products_path, task, task1_locales, fraction, seed, prod_dtypes)
    else:
        process_data(products_file, products_path, task, prod_dtypes)
        products = read_file(get_processed_file_name(products_file), products_path, task, task1_locales, fraction, seed, prod_dtypes)

    if check_processed(sessions_file, sessions_path):
        sessions = read_file(get_processed_file_name(sessions_file), sessions_path, task, task1_locales, fraction, seed, sess_dtypes)
    else:
        process_data(sessions_file, sessions_path, task, sess_dtypes)
        sessions = read_file(get_processed_file_name(sessions_file), sessions_path, task, task1_locales, fraction, seed, sess_dtypes)

    if check_processed(sessions_test_file, sessions_test_path):
        sessions_test = read_file(get_processed_file_name(sessions_test_file), sessions_test_path, task, task1_locales, fraction, seed, sess_dtypes)
    else:
        process_data(sessions_test_file, sessions_test_path, task, sess_dtypes)
        sessions_test = read_file(get_processed_file_name(sessions_test_file), sessions_test_path, task, task1_locales, fraction, seed, sess_dtypes)
    return products, sessions, sessions_test
