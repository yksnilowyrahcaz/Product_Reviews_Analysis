from logger import *
import csv, pathlib, time, pandas as pd
from reduce_memory_df import optimize_mem_usage

def preprocess(file_path: pathlib.Path) -> None:
    '''
    Preprocess Amazon US Customer Reviews Dataset tsv files.
    
    Performs the following steps:
    
    1. Read in file.
    2. Replace null values with ''.
    3. Combine review_headline with review_body.
    4. Drop unneeded columns.
    5. Optimize memory usage through data types.
    6. Save to parquet.
    '''
    logging.info('Reading file.')
    df = pd.read_csv(file_path, sep='\t', quoting=csv.QUOTE_NONE)

    logging.info('Filling null values.')
    df.fillna('', inplace=True)

    logging.info('Combining review body with headline.')
    df['review'] = (df['review_headline'] + ' ' + df['review_body']).str.strip()

    logging.info('Dropping uneeded columns.')
    df.drop(['marketplace', 'product_category', 'review_headline', 'review_body'], axis=1, inplace=True)

    logging.info('Optimizing memory usage.')
    optimize_mem_usage(df)

    logging.info('Saving to parquet.')
    df.to_parquet('data/' + file_path.stem + '_preprocessed.parquet')

    logging.info(f'Preprocessing complete.')

if __name__ == '__main__':
    for file_path in pathlib.Path.cwd().glob('data/*.tsv'):
        start = time.time()
        try:
            preprocess(file_path)
            logging.info(f'Total time: {(time.time() - start) / 60:,.2f} minutes')
        except Exception as e:
            logging.info(e.args)
            continue