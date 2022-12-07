import os, re, time, logging, umap, hdbscan, yake, pandas as pd
from utils import plot_embeddings, plot_clustered_embeddings, get_lex_fields
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

logging.basicConfig(
    level=logging.INFO,
    filename='logs/log.log',
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s'
    )

def log_and_print(msg):
    logging.info(msg)
    print(msg)

def pipeline(file_path):
    '''
    Embed a data set into two dimensions with UMAP 
    and cluster the embeddings using HDBSCAN. Then,
    label each cluster using YAKE key word extraction.

    param: file_path to data set
    return: None; instead, generates parquet static html 
            files saved to current working directory.
    '''
    log_and_print(f'Processing {file_path} ...')
    df = pd.read_parquet(f'cleaned_data/{file_path}')
    df = df[(df.verified_purchase == 'Y') & (df.helpful_votes/df.total_votes > 0.5)]
    sample = df.sample(min(100000, df.shape[0]), random_state=1729)
    sample['docs'] = df.product_title.str[:].copy() + ' ' + df.review.str[:].copy()    
    name = re.findall('(?<=reviews_)[.\w]*(?=.parq)', file_path)[0].lower()    

    log_and_print('tfidf vectorizing data ...')
    stop_words = ENGLISH_STOP_WORDS.union({'star','stars'})
    vectorizer = TfidfVectorizer(min_df=5, stop_words=stop_words)
    doc_term_matrix = vectorizer.fit_transform(sample.docs)

    log_and_print('learning UMAP embedding ...')
    embedding = umap.UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=2,
        metric='hellinger'
        ).fit(doc_term_matrix)

    sample['e1'] = embedding.embedding_[:,0]
    sample['e2'] = embedding.embedding_[:,1]

    log_and_print('learning HDBSCAN clusters ...')
    clusters = hdbscan.HDBSCAN(
        min_samples=10,
        min_cluster_size=500
        ).fit_predict(embedding.embedding_)

    log_and_print('learning YAKE cluster labels ...')
    sample['clusters'] = clusters
    kwd = {-1:'unclustered'}
    kwx = yake.KeywordExtractor(n=1, top=1)
    for cluster in sample[sample.clusters!=-1].groupby('clusters'):
        text = ' '.join(cluster[1].docs.tolist())
        kwd[cluster[0]] = kwx.extract_keywords(text)[0][0].lower()
    sample['labels'] = sample.clusters.map(kwd)

    log_and_print('generating plots and saving data ...')
    plot_embeddings(sample, name)
    plot_clustered_embeddings(sample, name)
    sample.drop('docs', axis=1, inplace=True)
    sample = sample.astype({col:'category' for col in ['clusters','labels']})
    sample.to_parquet(f'samples/labeled_samples/{name}_clustered_embeddings.parquet', index=False)

    log_and_print('getting sentiment lexical fields ...')
    return get_lex_fields(sample, name)

if __name__ == '__main__':
    dfs = []
    files = [file for file in os.listdir('cleaned_data')]
    for file in files:
        start = time.time()
        try:
            dfs.append(pipeline(file))
            print(f'Total time: {(time.time()-start)/60:,.2f} minutes')
        except Exception as e:
            logging.info(e.args)
            print('Exception encountered. See log for details. Continuing to next file.')
            continue 

    pd.concat(dfs).to_parquet('samples/data_for_power_bi/key_topics.parquet', index=False)
    print('Process Complete')