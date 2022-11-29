import pandas as pd
import os, logging, time

logging.basicConfig(
    filename='log.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s'
)

if __name__ == '__main__':
    files1 = [file for file in os.listdir('data')]
    files2 = [file for file in os.listdir('labeled_data')]

    for file1, file2 in zip(files1, files2):
        start = time.time()
        msg = f'merging {file1} with {file2}...'
        logging.info(msg)
        print(msg)

        df1 = pd.read_parquet(f'data/{file1}')
        df2 = pd.read_parquet(f'labeled_data/{file2}')
        df3 = df1.reset_index().merge(df2, left_on='index', right_on='_index')
        df3.drop(columns=['index','_index', 'review_y'], inplace=True)
        df3.rename(columns={'review_x':'review'}, inplace=True)
        name = file2.replace('_clustered_embeddings','_labeled_sample')
        df3.to_parquet(f'labeled_samples/{name}')

        msg = f'merge complete'
        logging.info(msg)
        print(msg)
        print(f'Total time: {(time.time()-start)/60:,.2f} minutes')