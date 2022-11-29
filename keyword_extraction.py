import pandas as pd
import os, logging, time, yake

logging.basicConfig(
    filename='log.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s'
)

def add_keyword_labels(df):
    kwd = {-1:'unclustered'}
    kwx = yake.KeywordExtractor(n=1, top=1)
    for cluster in df[df.cluster!=-1].groupby('cluster'):
        text = ' '.join(cluster[1].review.tolist())
        kwd[cluster[0]] = kwx.extract_keywords(text)[0][0].lower()
    df['label'] = df.cluster.map(kwd)
    return df

if __name__ == '__main__':
    files = [file for file in os.listdir('cluster_data')]
    for file in files:
        start = time.time()
        msg = f'labeling {file} ...'
        logging.info(msg)
        print(msg)

        df = pd.read_parquet(f'cluster_data/{file}')
        df = add_keyword_labels(df)
        df.to_parquet(f'labeled_data/{file}')

        msg = f'finished labeling, saved to labeled_data/{file}'
        logging.info(msg)
        print(msg)
        print(f'Total time: {(time.time()-start)/60:,.2f} minutes')