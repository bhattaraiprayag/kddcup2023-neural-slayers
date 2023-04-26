import pandas as pd
from typing import Tuple, Dict, List

# Function to analyze dataframes
def analyze_sessions(
    products_train: pd.DataFrame,
    sessions_train: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, Dict[str, int], Dict[str, List[int]]]:
    
    # 1. Identify unique product ids in products_train
    prod_in_pt = products_train['id'].unique()

    # 2. Identify unique product ids in sessions_train['prev_items']
    sessions_train = sessions_train.assign(prev_items=sessions_train['prev_items'].str.split(','))
    prod_in_st = sessions_train['prev_items'].explode().unique()

    # 3. Identify rows where each unique id in prod_in_pt can be found in products_train
    products_train = products_train.reset_index().rename(columns={'index': 'row'})
    prod_in_pt_rows = products_train[products_train['id'].isin(prod_in_pt)].set_index('id')['row'].to_dict()

    # 4. Identify rows where each unique id in prod_in_st occurs in sessions_train['prev_items'] and count occurrences
    exploded_sessions_train = sessions_train.explode('prev_items').reset_index().rename(columns={'index': 'row'})
    unique_pairs = exploded_sessions_train[['row', 'prev_items']].drop_duplicates()
    prod_in_st_occs = unique_pairs[unique_pairs['prev_items'].isin(prod_in_st)].groupby('prev_items')['row'].apply(list).to_dict()
    
    return prod_in_pt, prod_in_st, prod_in_pt_rows, prod_in_st_occs

# Function to view common products
def view_common_products(
    prod_pt: pd.Series,
    prod_st: pd.Series,
    prod_pt_rows: Dict[str, int],
    prod_st_occs: Dict[str, List[int]]
) -> pd.DataFrame:

    # Calculate common product ids
    common_ids = set(prod_pt) & set(prod_st)

    app_rows = []

    for common_id in common_ids:
        count_in_sessions = len(prod_st_occs[common_id])
        row_in_products = prod_pt_rows[common_id] + 1
        app_data = prod_st_occs[common_id]

        app_rows.append([common_id, row_in_products, ",".join(map(str, [x + 1 for x in app_data])), count_in_sessions])

    # Create a DataFrame to store the results
    app_df = pd.DataFrame(app_rows, columns=['id', 'row_prod', 'row_sess', 'count_sess'])
    app_df = app_df.sort_values(by=['row_prod']).reset_index(drop=True)
    
    return app_df

# Function to split a dataset based on locales
def split_locales(df: pd.DataFrame, locales: List[str]) -> List[pd.DataFrame]:
    return [df[df['locale'] == locale] for locale in locales]

# Function to split a dataset based on locales and save to csv
def split_locales_and_save(df: pd.DataFrame, locales: List[str], output_dir: str, filename: str) -> None:
    for locale in locales:
        df[df['locale'] == locale].to_csv(f"{output_dir}/{filename}_{locale}.csv", index=False)

# Function to split sessions into train, val and test sets
def split_sessions(df: pd.DataFrame, val_size: float, test_size: float) -> List[pd.DataFrame]:
    # Check if the sizes add up to 1
    train_size = 1 - (val_size + test_size)
    assert train_size >= 0, "The sum of val_size and test_size must be less than or equal to 1"
    
    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Split the dataframe into train, val and test sets
    train_df = df.iloc[:int(len(df) * train_size)]
    val_df = df.iloc[int(len(df) * train_size):int(len(df) * (train_size + val_size))]
    test_df = df.iloc[int(len(df) * (train_size + val_size)):]
    
    return [train_df, val_df, test_df]