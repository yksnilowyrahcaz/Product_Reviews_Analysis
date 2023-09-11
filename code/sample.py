from logger import *
from pathlib import Path
import dask
import pandas as pd
import time
import traceback

def sample_reviews(file_path: Path) -> tuple[pd.DataFrame]:
    '''
    Uniformly sample from sample key topics
    '''
    df = pd.read_parquet(file_path)
    df = df[df['clusters'] != -1]

    def reformat_review_text(df):
        df = df.sample(min(5000, df.shape[0]), random_state=1729)
        df['product_topic'] = df['product_category'].str.lower() + ' ' + df['topics']
        df['text'] = 'Rating: ' + df['star_rating'].astype(str) + ', ' + \
            'Title: ' + df['product_title'].astype(str) + ', ' + \
                'Review: ' + df['review']
        return df

    bad = reformat_review_text(df[df['star_rating'] <= 3])
    good = reformat_review_text(df[df['star_rating'] > 3])
    cols = ['product_topic', 'product_category', 'topics', 'text']

    return bad[cols], good[cols]

if __name__ == '__main__':
    start = time.time()
    logging.info('Sampling reviews.')

    tasks = [
        dask.delayed(sample_reviews)(file)
        for file in Path.cwd().glob('data/*key_topics.parquet')
    ]

    try:
        results = dask.delayed()(tasks).compute()
        pd.concat(
            [result[0] for result in results]
        ).to_parquet(
            'data/bad_reviews_sample.parquet',
            index=False
        )
        pd.concat(
            [result[1] for result in results]
        ).to_parquet(
            'data/good_reviews_sample.parquet',
            index=False
        )
        logging.info(f'Total time: {(time.time() - start) / 60:,.2f} minutes')

    except:
        logging.exception(traceback.format_exc())