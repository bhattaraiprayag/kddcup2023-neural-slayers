import os
import re
import string
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from typing import List, Tuple, Union

# Main function to handle data processing
def handle_data(
    project_files: List[str],
    file_paths: List[str],
    task: str,
    task1_locales: List[str],
    num_partitions: int,
    partition_ids: dict,
    fraction: float,
    seed: int,
    prod_dtypes: dict,
    sess_dtypes: dict
) -> Tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame]:

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
        num_partitions: int,
        partition_id: Union[str, int, List[int], range],
        fraction: float,
        seed: int,
        dtypes: dict
    ) -> dd.DataFrame:
        file_path = os.path.join(path, file)
        df = dd.read_csv(file_path, dtype=dtypes, assume_missing=True)
        df = df[df['locale'].isin(task1_locales)]

        # Filter data based on partition and fraction settings
        if num_partitions == None:
            if fraction < 1:
                df = df.sample(frac=fraction, random_state=seed)
        elif isinstance(num_partitions, int):
            df = df.repartition(npartitions=num_partitions)
            if partition_id != 'all':
                if isinstance(partition_id, (int, list)):
                    df = df.partitions[partition_id]
                    if fraction < 1:
                        df = df.sample(frac=fraction, random_state=seed)
                elif isinstance(partition_id, range):
                    df = df.partitions[partition_id[0]:partition_id[-1]+1]
                    if fraction < 1:
                        df = df.sample(frac=fraction, random_state=seed)
            elif partition_id == 'all':
                if fraction < 1:
                    df = df.sample(frac=fraction, random_state=seed)
            else:
                raise ValueError('Invalid partition_id')
        else:
            raise ValueError('Invalid num_partitions')

        return df

    # Function to process data and save the processed file
    def process_data(file: str, path: str, task: str, dtypes: dict) -> None:
        file_path = os.path.join(path, file)
        processed_file = os.path.join(path, file.split('.')[0] + '_processed.csv')
        df = dd.read_csv(file_path, dtype=dtypes, assume_missing=True)

        # Process products_train.csv file
        if 'products_train.csv' in file:
            non_text_cols = ['id', 'locale', 'price']
            text_cols = [col for col in df.columns if col not in non_text_cols]
            df[text_cols] = df[text_cols].fillna('n/a')

            # Define characters to preserve and create patterns to remove unwanted characters
            preserve_english = string.ascii_letters + string.digits
            preserve_german = 'äöüÄÖÜß'
            preserve_japanese = r'\u3041-\u3096\u3099\u309A\u30A0-\u30FF\u3400-\u4DB5\u4E00-\u9FEC\uF900-\uFA6B'
            preserve_french = 'àâæçèéêëîïôœùûüÿÀÂÆÇÈÉÊËÎÏÔŒÙÛÜŸ'
            preserve_spanish = 'áéíóúýÁÉÍÓÚÝñÑ¿¡'
            patterns_to_preserve = preserve_english + preserve_german + preserve_japanese + preserve_french + preserve_spanish

            pattern = [
                (r'<.*?>', ''),
                (r'[^\[' + patterns_to_preserve + r'\]]+', ' '),
                (r'\s+', ' '),
                (r'^\s+|\s+?$', '')
            ]

            # Clean text columns by removing unwanted characters and converting to lowercase
            for col in text_cols:
                for pat, repl in pattern:
                    df[col] = df[col].str.replace(pat, repl, regex=True)
                    df[col] = df[col].str.lower()

        # Process sessions_train.csv and sessions_test.csv files
        elif 'sessions_train.csv' in file or f'sessions_test_{task}.csv' in file:
            df['session_id'] = df.index + 1
            df['session_id'] = df['session_id'].astype('int32')
            df = df[[df.columns[-1]] + list(df.columns[:-1])]
            df['prev_items'] = df['prev_items'].str.findall(r"'(\w+)'").apply(lambda x: ','.join(x), meta=('prev_items', 'object'))

        # Save processed file with ProgressBar
        with ProgressBar():
            try:
                print(f'Saving {processed_file.split("/")[-1]} to path: {path}.')
                df.to_csv(processed_file, single_file=True, index=False, compute=True)
                print(f'{processed_file.split("/")[-1]} saved successfully to path: {path}.')
            except Exception as e:
                print(f'Error saving processed file: {processed_file.split("/")[-1]} to path: {path}.')
                print(e)

    products_file, sessions_file, sessions_test_file = project_files
    products_path, sessions_path, sessions_test_path = file_paths

    partition_ids = {
        'products_train': partition_ids['products_train'],
        'sessions_train': partition_ids['sessions_train'],
        'sessions_test': partition_ids['sessions_test']
    }

    # Check if processed files exist and load them, otherwise process data and save processed files
    if check_processed(products_file, products_path):
        print(f'Processed products file found at: {products_path}. Loading it now...')
        products = read_file(get_processed_file_name(products_file), products_path, task, task1_locales, num_partitions, partition_ids['products_train'], fraction, seed, prod_dtypes)
    else:
        print(f'Processed products file not found at: {products_path}.')
        process_data(products_file, products_path, task, prod_dtypes)
        print(f'Processed products file saved at: {products_path}.')
        products = read_file(get_processed_file_name(products_file), products_path, task, task1_locales, num_partitions, partition_ids['products_train'], fraction, seed, prod_dtypes)

    if check_processed(sessions_file, sessions_path):
        print(f'Processed sessions file found at: {sessions_path}. Loading it now...')
        sessions = read_file(get_processed_file_name(sessions_file), sessions_path, task, task1_locales, num_partitions, partition_ids['sessions_train'], fraction, seed, sess_dtypes)
    else:
        print(f'Processed sessions file not found at: {sessions_path}.')
        process_data(sessions_file, sessions_path, task, sess_dtypes)
        print(f'Processed sessions file saved at: {sessions_path}.')
        sessions = read_file(get_processed_file_name(sessions_file), sessions_path, task, task1_locales, num_partitions, partition_ids['sessions_train'], fraction, seed, sess_dtypes)

    if check_processed(sessions_test_file, sessions_test_path):
        print(f'Processed sessions test file found at: {sessions_test_path}. Loading it now...')
        sessions_test = read_file(get_processed_file_name(sessions_test_file), sessions_test_path, task, task1_locales, num_partitions, partition_ids['sessions_test'], fraction, seed, sess_dtypes)
    else:
        print(f'Processed sessions test file not found at: {sessions_test_path}.')
        process_data(sessions_test_file, sessions_test_path, task, sess_dtypes)
        print(f'Processed sessions test file saved at: {sessions_test_path}.')
        sessions_test = read_file(get_processed_file_name(sessions_test_file), sessions_test_path, task, task1_locales, num_partitions, partition_ids['sessions_test'], fraction, seed, sess_dtypes)

    # Return processed dataframes
    return products, sessions, sessions_test