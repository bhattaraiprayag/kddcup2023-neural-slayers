import os
import pandas as pd


# Function to load and clean products data
def load_and_clean_data(file_path, is_products=False, processed_file=None, slice_size=None):
    """
    This function loads and cleans data from the given file_path.
    If the is_products flag is set to True, it fills missing values and removes special characters from the data.
    If a processed file is specified and exists, it loads the data from there instead of the original file.
    If a slice size is specified, it loads only a portion of the data.
    """
    if is_products and processed_file and os.path.exists(processed_file):
        df = pd.read_csv(processed_file, nrows=slice_size) if slice_size else pd.read_csv(processed_file)
    else:
        df = pd.read_csv(file_path, nrows=slice_size) if slice_size else pd.read_csv(file_path)

        if is_products:
            # Fill missing values
            df.fillna('unknown', inplace=True)

            # Remove special characters
            pattern = [
                (r'<.*?>', ''),                             # Remove HTML tags
                (r'[^\x00-\x7F]+', ' '),                    # Remove non-ASCII characters
                (r'[^a-zA-Z0-9\s]', ''),                    # Remove special characters
                (r'\s+', ' '),                              # Remove extra spaces
                (r'^\s+|\s+?$', ''),                        # Remove leading and trailing spaces
            ]

            # Clean text
            for col in df.select_dtypes('object').columns:
                if col not in non_text_cols:
                    text = df[col].str.lower()
                    for p, r in pattern:
                        text = text.str.replace(p, r, regex=True)
                    df[col] = text

            if processed_file and not os.path.exists(processed_file):
                df.to_csv(processed_file, index=False)

    return df


# Set up environment
task = 'task1'
train_path = '../../data/train/'
test_path = '../../data/test/sessions_test_' + task + '.csv'
PREDS_PER_SESSION = 100
# slice_size = 10000                                           # Memory management, None for entire dataset
non_text_cols = ['id', 'locale', 'price']


# Function to load all data
def load_all_data(slice_size=None):
    """
    This function loads and cleans all necessary data for the project.
    It calls the load_and_clean_data function for each dataset and returns them as a tuple.
    """
    # Load and clean data
    processed_file = train_path + '/products_train_processed.csv'
    products_train = load_and_clean_data(train_path + '/products_train.csv', is_products=True, processed_file=processed_file, slice_size=slice_size)
    sessions_train = load_and_clean_data(train_path + '/sessions_train.csv', slice_size=slice_size)
    sessions_test = load_and_clean_data(test_path, slice_size=slice_size)

    return products_train, sessions_train, sessions_test
