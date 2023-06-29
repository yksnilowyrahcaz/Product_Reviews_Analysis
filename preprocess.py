from logger import *
from pathlib import Path
import csv, time, pandas as pd
from reduce_df_memory import optimize_memory_usage

def preprocess(file_path: Path) -> None:
    '''
    Preprocess Amazon US Customer Reviews Dataset tsv files.
    
    Performs the following steps:
    
    1. Read in file.
    2. Replace null values with ''.
    3. Combine review_headline with review_body.
    4. Drop unneeded columns.
    5. Optimize memory usage through data types.
    6. Save to parquet.
    
    :param file_path: path to the tsv file
    :type file_path: pathlib.Path
    
    :return: None
    '''
    logging.info('Reading file.')
    df = pd.read_csv(file_path, sep='\t', quoting=csv.QUOTE_NONE)

    logging.info('Filling null values.')
    df.fillna('', inplace=True)

    logging.info('Combining review body with headline.')
    df['review'] = (df['review_headline'] + ' ' + df['review_body']).str.strip()

    logging.info('Dropping uneeded columns.')
    df.drop(['marketplace', 'review_headline', 'review_body'], axis=1, inplace=True)

    logging.info('Optimizing memory usage.')
    optimize_memory_usage(df)

    logging.info('Saving to parquet.')
    df.to_parquet('data/' + file_path.stem + '_preprocessed.parquet')

    logging.info(f'Preprocessing complete.')

if __name__ == '__main__':
    for file_path in Path.cwd().glob('data/amazon_reviews_us_Gift_Card_v1_00.tsv'):
        start = time.time()
        try:
            preprocess(file_path)
            logging.info(f'Total time: {(time.time() - start) / 60:,.2f} minutes')
        except Exception as e:
            logging.exception(e.args)
            continue