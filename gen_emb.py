import os
import faiss
import numpy as np
from IPython.display import clear_output
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Function to create product embeddings
def create_prod_embeddings(products_train, combined_features, output_path, batch_size):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = []
    texts_to_embed = products_train[combined_features].astype(str).agg(' '.join, axis=1).tolist()
    num_batches = len(texts_to_embed) // batch_size + (1 if len(texts_to_embed) % batch_size != 0 else 0)
    for i in tqdm(range(num_batches), desc="Creating Embeddings..."):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(texts_to_embed))
        batch_texts = texts_to_embed[start:end]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False, batch_size=batch_size)
        embeddings.append(batch_embeddings)
    np.save(output_path, np.vstack(embeddings))

# Function to load locale embeddings
def load_locale_embeddings(locale, products, combined_features, batch_size, locale_embed_path):
    print(locale_embed_path)
    print(type(locale_embed_path))
    print(f"Loading embeddings for locale: {locale}")
    if os.path.exists(locale_embed_path[locale]):
        embeddings = np.load(locale_embed_path[locale])
        faiss.normalize_L2(embeddings)
    else:
        create_prod_embeddings(products, combined_features, locale_embed_path[locale], batch_size)
        embeddings = np.load(locale_embed_path[locale])
        faiss.normalize_L2(embeddings)
    return embeddings
