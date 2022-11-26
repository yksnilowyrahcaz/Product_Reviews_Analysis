import pandas as pd
import os, re, csv, time, logging
from reduce_memory_df import optimize_mem_usage

logging.basicConfig(
    level=logging.INFO,
    filename='log.log',
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s'
    )

def preprocess(file_path):
    msg = f'Processing {file_path} ...'
    logging.info(msg)
    print(msg)

    print('reading file ...')
    df = pd.read_csv(f'data/{file_path}', sep='\t', quoting=csv.QUOTE_NONE)

    print('filling null values ...')
    df.fillna('', inplace=True)

    print('combining review body with headline ...')
    df['review'] = (df.review_headline + ' ' + df.review_body).str.strip()

    print('dropping uneeded columns ...')
    df.drop(['marketplace','product_category','review_headline','review_body'], axis=1, inplace=True)

    print('optimizing memory usage ...')
    optimize_mem_usage(df, ignore=['review_id','review'])

    print('saving to parquet ...')
    name = re.findall('(?<=us_)[.\w]*(?=_v1)', file_path)[0].lower()
    df.to_parquet(f'data/amazon_reviews_{name}.parquet')

    msg = f'Processing complete. File saved to data/amazon_reviews_{name}.parquet'
    logging.info(msg)
    print(msg)

if __name__ == '__main__':
    files = [file for file in os.listdir('data') if 'tsv' in file]
    for file in files:
        start = time.time()
        try:
            preprocess(file)
            print(f'Total time: {(time.time()-start)/60:,.2f} minutes')
        except Exception as e:
            logging.info(e.args)
            print('Exception encountered. See log for details. Continuing to next file.')
            continue