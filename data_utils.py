# data_utils.py

import pandas as pd
import numpy as np

from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler


def analyze_sessions(
    products_train: pd.DataFrame,
    sessions_train: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, Dict[str, int], Dict[str, List[int]]]:
    prod_in_pt = products_train['id'].unique()
    sessions_train = sessions_train.assign(prev_items=sessions_train['prev_items'].str.split(','))
    prod_in_st = sessions_train['prev_items'].explode().unique()
    products_train = products_train.reset_index().rename(columns={'index': 'row'})
    prod_in_pt_rows = products_train[products_train['id'].isin(prod_in_pt)].set_index('id')['row'].to_dict()
    exploded_sessions_train = sessions_train.explode('prev_items').reset_index().rename(columns={'index': 'row'})
    unique_pairs = exploded_sessions_train[['row', 'prev_items']].drop_duplicates()
    prod_in_st_occs = unique_pairs[unique_pairs['prev_items'].isin(prod_in_st)].groupby('prev_items')['row'].apply(list).to_dict()
    return prod_in_pt, prod_in_st, prod_in_pt_rows, prod_in_st_occs


def view_common_products(
    prod_pt: pd.Series,
    prod_st: pd.Series,
    prod_pt_rows: Dict[str, int],
    prod_st_occs: Dict[str, List[int]]
) -> pd.DataFrame:
    common_ids = set(prod_pt) & set(prod_st)
    app_rows = []
    for common_id in common_ids:
        count_in_sessions = len(prod_st_occs[common_id])
        row_in_products = prod_pt_rows[common_id] + 1
        app_data = prod_st_occs[common_id]
        app_rows.append([common_id, row_in_products, ",".join(map(str, [x + 1 for x in app_data])), count_in_sessions])
    app_df = pd.DataFrame(app_rows, columns=['id', 'row_prod', 'row_sess', 'count_sess'])
    app_df = app_df.sort_values(by=['row_prod']).reset_index(drop=True)
    return app_df


def split_locales(df: pd.DataFrame, locales: List[str]) -> List[pd.DataFrame]:
    if 'title' in df.columns:          # products
        rewrite_col = None             # DO NOT modify `id`
    elif 'prev_items' in df.columns:   # sessions
        rewrite_col = 'session_id'
    else:
        raise ValueError("Unexpected dataframe schema")
    out = []
    for loc in locales:
        df_loc = df[df['locale'] == loc].copy()
        if rewrite_col and not df_loc.empty:
            df_loc[rewrite_col] = np.arange(1, len(df_loc) + 1,
                                            dtype=df_loc[rewrite_col].dtype)
        out.append(df_loc)
    return out


def split_locales_and_save(df: pd.DataFrame, locales: List[str], output_dir: str, filename: str) -> None:
    for locale in locales:
        df[df['locale'] == locale].to_csv(f"{output_dir}/{filename}_{locale}.csv", index=False)


def split_sessions(df: pd.DataFrame, val_size: float, test_size: float) -> List[pd.DataFrame]:
    train_size = 1 - (val_size + test_size)
    assert train_size >= 0, "The sum of val_size and test_size must be less than or equal to 1"
    df = df.sample(frac=1).reset_index(drop=True)
    train_df = df.iloc[:int(len(df) * train_size)]
    val_df = df.iloc[int(len(df) * train_size):int(len(df) * (train_size + val_size))]
    test_df = df.iloc[int(len(df) * (train_size + val_size)):]
    return [train_df, val_df, test_df]


def scale_prices(df, locales):
    scaled_df = df.copy()
    for locale in locales:
        scaler = StandardScaler()
        locale_mask = scaled_df['locale'] == locale
        if not scaled_df[locale_mask].empty:
            scaled_df.loc[locale_mask, 'price'] = scaler.fit_transform(scaled_df.loc[locale_mask, ['price']])
        else:
            print(f"No data found for locale: {locale}")
    return scaled_df
