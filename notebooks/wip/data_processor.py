import os
import pandas as pd
import re

def load_and_process_data(user_path, products_slice=None, sessions_slice=None, test_slice=None, task=None):
    # Function to process products_train.csv
    def process_products(df):
        # Fill missing values with 'unknown'
        df.fillna('unknown', inplace=True)
        
        # Define patterns for cleaning text
        pattern = [(r'<.*?>', ''), (r'[^\x00-\x7F]+', ' '), (r'[^a-zA-Z0-9\s]', ''),
                   (r'\s+', ' '), (r'^\s+|\s+?$', '')]
        
        # Columns that are not text
        non_text_cols = ['id', 'locale', 'price']
        
        # Clean text columns
        for col in df.select_dtypes('object').columns:
            if col not in non_text_cols:
                text = df[col].str.lower()
                for p, r in pattern:
                    text = text.str.replace(p, r, regex=True)
                df[col] = text
        return df

    # Function to process sessions_train.csv and sessions_test.csv
    def process_sessions(df):
        # # Add session_id column as the first column
        # df['session_id'] = df.index + 1
        # df['session_id'] = df['session_id'].astype('int32')
        # df = df[[df.columns[-1]] + list(df.columns[:-1])]
        
        # Extract and join product IDs from prev_items
        df['prev_items'] = df['prev_items'].str.findall(r"'(\w+)'").apply(lambda x: ','.join(x))
        return df

    # Function to load and process a CSV file
    def load_process_csv(path, filename, processor=None, slice=None):
        processed_file = os.path.join(path, f'{os.path.splitext(filename)[0]}_processed.csv')
        
        # Check if processed file exists
        if os.path.exists(processed_file):
            df = pd.read_csv(processed_file, nrows=slice)
        else:
            # Load and process the file if it doesn't exist
            df = pd.read_csv(os.path.join(path, filename), nrows=slice)
            if processor: df = processor(df)
            
            # Save processed file
            df.to_csv(processed_file, index=False)
        return df

    # Define train and test paths
    train_path, test_path = os.path.join(user_path, 'train'), os.path.join(user_path, 'test')
    
    # Define train files and corresponding processors
    train_files = [('products_train.csv', process_products, products_slice), ('sessions_train.csv', process_sessions, sessions_slice)]
    
    # Define test file if a task is specified
    test_file = f'sessions_test_{task}.csv' if task else None

    # Load and process train files
    products_train, sessions_train = (load_process_csv(train_path, *file) for file in train_files)
    
    # Load and process test file if specified
    sessions_test = load_process_csv(test_path, test_file, process_sessions, test_slice) if test_file else None

    return products_train, sessions_train, sessions_test
