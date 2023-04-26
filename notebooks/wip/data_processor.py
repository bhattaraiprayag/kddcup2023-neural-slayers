import os
import re
import pandas as pd
from typing import List, Tuple

# Main function to handle data processing
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

    # Function to check if a processed file exists
    def check_processed(file: str, path: str) -> bool:
        processed_file = os.path.join(path, file.split('.')[0] + '_processed.csv')
        return os.path.exists(processed_file)

    # Function to get the processed file name
    def get_processed_file_name(file: str) -> str:
        return file.split('.')[0] + '_processed.csv'

    # Function to read a file and filter it based on input parameters
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

    # Function to process data and save the processed file
    def process_data(file: str, path: str, task: str, dtypes: dict) -> None:
        file_path = os.path.join(path, file)
        processed_file = os.path.join(path, file.split('.')[0] + '_processed.csv')
        df = pd.read_csv(file_path, dtype=dtypes)

        # Process products_train.csv file
        if 'products_train.csv' in file:
            # Replace string representations of missing values with actual NaN values
            df.replace(["Nan", "nan", "Null", "N/a", "NaN", "n/A"], pd.NA, inplace=True)

            # Replace NaN values with empty strings
            df.fillna('-', inplace=True)

            # Identify columns to process
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
        # Process sessions_train.csv and sessions_test.csv files
        elif 'sessions_train.csv' in file or f'sessions_test_{task}.csv' in file:
            df['session_id'] = df.index + 1
            df['session_id'] = df['session_id'].astype('int32')
            df = df[[df.columns[-1]] + list(df.columns[:-1])]
            df['prev_items'] = df['prev_items'].str.findall(r"'(\w+)'").apply(lambda x: ','.join(x))
            df = df[['session_id', 'locale', 'prev_items', 'next_item']] if 'sessions_train.csv' in file else df[['session_id', 'locale', 'prev_items']]
        
        # Save processed file
        try:
            print(f'Saving {processed_file.split("/")[-1]} to: {path}.')
            df.to_csv(processed_file, index=False)
            print(f'Saved {processed_file.split("/")[-1]} to: {path}.')
        except Exception as e:
            print(f'Error saving processed file: {processed_file.split("/")[-1]} to: {path}.')
            print(e)

    products_file, sessions_file, sessions_test_file = project_files
    products_path, sessions_path, sessions_test_path = file_paths

    # Check if processed files exist and load them, otherwise process data and save processed files, then load them
    if check_processed(products_file, products_path):
        print(f'Processed products file found at: {products_path} Loading it now...')
        products = read_file(get_processed_file_name(products_file), products_path, task, task1_locales, fraction, seed, prod_dtypes)
    else:
        print(f'Processed products file not found at: {products_path}')
        process_data(products_file, products_path, task, prod_dtypes)
        products = read_file(get_processed_file_name(products_file), products_path, task, task1_locales, fraction, seed, prod_dtypes)

    if check_processed(sessions_file, sessions_path):
        print(f'Processed sessions file found at: {sessions_path} Loading it now...')
        sessions = read_file(get_processed_file_name(sessions_file), sessions_path, task, task1_locales, fraction, seed, sess_dtypes)
    else:
        print(f'Processed sessions file not found at: {sessions_path}')
        process_data(sessions_file, sessions_path, task, sess_dtypes)
        sessions = read_file(get_processed_file_name(sessions_file), sessions_path, task, task1_locales, fraction, seed, sess_dtypes)

    if check_processed(sessions_test_file, sessions_test_path):
        print(f'Processed sessions test file found at: {sessions_test_path} Loading it now...')
        sessions_test = read_file(get_processed_file_name(sessions_test_file), sessions_test_path, task, task1_locales, fraction, seed, sess_dtypes)
    else:
        print(f'Processed sessions test file not found at: {sessions_test_path}')
        process_data(sessions_test_file, sessions_test_path, task, sess_dtypes)
        sessions_test = read_file(get_processed_file_name(sessions_test_file), sessions_test_path, task, task1_locales, fraction, seed, sess_dtypes)

    # Return processed dataframes
    return products, sessions, sessions_test