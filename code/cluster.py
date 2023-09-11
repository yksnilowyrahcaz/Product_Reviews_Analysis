from logger import *
from pathlib import Path
from sklearn.feature_extraction.text import (
    TfidfVectorizer, 
    ENGLISH_STOP_WORDS
)
from utils import (
    plot_embeddings,
    get_sentiment_keywords,
    plot_clustered_embeddings
)
import hdbscan
import pandas as pd
import time
import traceback
import umap
import yake

def run_pipeline(file_path: Path) -> pd.DataFrame:
    '''
    Does the following:
    
    1. Filter on reviews having a verified purchase
    and helpful votes ratio > 50%.
    
    2. Uniformly sample at most 100,000 reviews per dataset.
    
    3. TFIDF vectorize the reviews into a document-term matrix.
    
    4. Embed the document-term matrix in two dimensions with UMAP.
    
    5. Cluster the embeddings using HDBSCAN.
    
    6. Extract the keyword for each cluster using YAKE
    to identify key topics.
    
    7. Separate bad (1-3 stars) and good (4-5 stars) 
    reviews and extract keywords by key topic.

    :param file_path: file_path to preprocessed data file
    :type file_path: pathlib.Path
    
    :return: pd.DataFrame
    '''
    logging.info(f'Reading {file_path.name}')
    df = pd.read_parquet(file_path)
    
    logging.info('Sampling reviews.')
    df = df[
        (df['verified_purchase'] == 'Y') & \
        (df['helpful_votes'] / df['total_votes'] > 0.5)
    ]
    sample = df.sample(min(100_000, df.shape[0]), random_state=1729)
    sample['docs'] = df['product_title'].str[:] + ' ' + df['review'].str[:]
    dataset_name = 'Multilingual' if 'multilingual' in file_path.name \
        else sample['product_category'].iloc[0].replace('_', ' ').title()

    logging.info('TFIDF vectorizing data.')
    stop_words = list(
        ENGLISH_STOP_WORDS.union({
            'star', 'stars', *dataset_name.lower().split()
        })
    )
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.1, stop_words=stop_words)
    doc_term_matrix = vectorizer.fit_transform(sample['docs'])

    logging.info('Learning UMAP embeddings.')
    embedding = umap.UMAP(
        n_neighbors=30,
        n_components=2,
        metric='hellinger',
        min_dist=0.0
    ).fit(doc_term_matrix)
    sample['e1'] = embedding.embedding_[:, 0]
    sample['e2'] = embedding.embedding_[:, 1]

    logging.info('Learning HDBSCAN clusters.')
    clusters = hdbscan.HDBSCAN(
        min_samples=10,
        min_cluster_size=500
    ).fit_predict(embedding.embedding_)
    sample['clusters'] = clusters

    logging.info('Learning YAKE cluster topics.')
    kwd = {-1: 'unclustered'}
    kwx = yake.KeywordExtractor(n=1, top=1, stopwords=stop_words)
    for cluster in sample[sample['clusters'] != -1].groupby('clusters'):
        text = ' '.join(cluster[1]['docs'].tolist())
        kwd[cluster[0]] = kwx.extract_keywords(text)[0][0].lower()
    sample['topics'] = sample['clusters'].map(kwd)

    logging.info('Generating plots and saving key topics.')
    plot_embeddings(sample, dataset_name)
    plot_clustered_embeddings(sample, dataset_name)
    sample.drop(columns='docs', inplace=True)
    sample.to_parquet(
        f'data/{file_path.stem.replace("preprocessed", "sample")}_key_topics.parquet',
        index=False
    )
    logging.info('Learning and saving topic keywords.')
    sentiment_df = get_sentiment_keywords(sample, dataset_name, stop_words)
    sentiment_df.to_parquet(
        f'data/{file_path.stem.replace("preprocessed", "sample")}_keywords.parquet',
        index=False
    )
    return sentiment_df

if __name__ == '__main__':
    dfs = []
    for file_path in Path.cwd().glob('data/*preprocessed.parquet'):
        start = time.time()
        try:
            dfs.append(run_pipeline(file_path))
            logging.info(f'Total time: {(time.time() - start) / 60:,.2f} minutes')
        except:
            logging.exception(traceback.format_exc())
            continue

    pd.concat(dfs).to_parquet('data/combined_keywords.parquet', index=False)
    logging.info('Process complete.')