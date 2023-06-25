from logger import *
import pathlib, time, yake, pandas as pd

def add_keyword_labels(df):
    kwd = {-1: 'unclustered'}
    kwx = yake.KeywordExtractor(n=1, top=1)
    for cluster in df[df.cluster!=-1].groupby('cluster'):
        text = ' '.join(cluster[1].review.tolist())
        kwd[cluster[0]] = kwx.extract_keywords(text)[0][0].lower()
    df['label'] = df.cluster.map(kwd)
    return df

if __name__ == '__main__':
    for file_path in pathlib.Path.cwd().glob('data/*.tsv'):
        start = time.time()
        
        logging.info(f'Reading file.')
        df = pd.read_parquet(f'cluster_data/{file}')
        
        logging.info('Labeling data.')
        df = add_keyword_labels(df)
        
        logging.info('Saving labeled data.')
        df.to_parquet(f'labeled_data/{file}')

        logging.info(f'Total time: {(time.time()-start) / 60:,.2f} minutes')