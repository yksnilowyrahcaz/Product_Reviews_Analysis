from logger import *
from pathlib import Path
import time, umap, hdbscan, yake, pandas as pd
from utils import plot_embeddings, plot_clustered_embeddings, get_lex_fields
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

def run_pipeline(file_path: Path) -> pd.DataFrame:
    '''
    Embed a data set into two dimensions with UMAP 
    and cluster the embeddings using HDBSCAN. Then,
    label each cluster using YAKE key word extraction.

    :param file_path: file_path to preprocessed data file
    :type file_path: pathlib.Path
    
    :return: pd.DataFrame
    '''
    logging.info('Reading file.')
    df = pd.read_parquet(file_path)
    df = df[(df['verified_purchase'] == 'Y') & (df['helpful_votes'] / df['total_votes'] > 0.5)]
    sample = df.sample(min(100_000, df.shape[0]), random_state=1729)
    sample['docs'] = df['product_title'].str[:].copy() + ' ' + df['review'].str[:].copy()

    logging.info('TFIDF vectorizing data.')
    stop_words = ENGLISH_STOP_WORDS.union({'star','stars'})
    vectorizer = TfidfVectorizer(min_df=5, stop_words=stop_words)
    doc_term_matrix = vectorizer.fit_transform(sample.docs)

    logging.info('Learning UMAP embedding.')
    embedding = umap.UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=2,
        metric='hellinger'
    ).fit(doc_term_matrix)

    sample['e1'] = embedding.embedding_[:,0]
    sample['e2'] = embedding.embedding_[:,1]

    logging.info('Learning HDBSCAN clusters.')
    clusters = hdbscan.HDBSCAN(
        min_samples=10,
        min_cluster_size=500
    ).fit_predict(embedding.embedding_)

    logging.info('Learning YAKE cluster labels.')
    sample['clusters'] = clusters
    kwd = {-1:'unclustered'}
    kwx = yake.KeywordExtractor(n=1, top=1)
    for cluster in sample[sample.clusters!=-1].groupby('clusters'):
        text = ' '.join(cluster[1].docs.tolist())
        kwd[cluster[0]] = kwx.extract_keywords(text)[0][0].lower()
    sample['labels'] = sample.clusters.map(kwd)

    logging.info('Generating plots and saving data.')
    plot_embeddings(sample, file_path.stem)
    plot_clustered_embeddings(sample, file_path.stem)
    sample.drop('docs', axis=1, inplace=True)
    sample = sample.astype({col:'category' for col in ['clusters', 'labels']})
    Path.mkdir(Path(Path.cwd(), 'samples/labeled_samples'), exist_ok=True)
    sample.to_parquet(f'samples/labeled_samples/{name}_clustered_embeddings.parquet', index=False)

    logging.info('Getting sentiment lexical fields.')
    return get_lex_fields(sample, file_path.stem)

if __name__ == '__main__':
    dfs = []
    for file_path in Path.cwd().glob('data/*.parquet'):
        start = time.time()
        try:
            dfs.append(run_pipeline(file_path))
            logging.info(f'Total time: {(time.time()-start)/60:,.2f} minutes')
        except Exception as e:
            logging.info(e.args)
            logging.info('Exception encountered. See log for details. Continuing to next file.')
            continue 

    pd.concat(dfs).to_parquet('samples/data_for_power_bi/key_topics.parquet', index=False)
    logging.info('Process Complete')