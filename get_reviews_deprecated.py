import os, logging, time, pandas as pd

logging.basicConfig(
    level=logging.INFO,
    filename='log.log',
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s'
    )

def log_and_print(msg):
    logging.info(msg)
    print(msg)

key_topics_df = pd.read_parquet('samples/reviews_key_topics.parquet')

def get_reviews_having_keywords(file):
    '''
    Iterate over labeled samples and extract just the reviews 
    that contain the key words identified by YAKE algorithm.
    '''
    df = pd.read_parquet(f'samples/labeled_samples_v3/{file}')
    category = df.product_category.unique()[0]
    ktops = key_topics_df[key_topics_df.product_category==category]

    bad_reviews = []
    good_reviews = []
    for label in ktops.labels.unique():
        if label != 'unclustered':
            bad_words = ktops[ktops.labels==label].bad_set.iloc[0].split()
            good_words = ktops[ktops.labels==label].good_set.iloc[0].split()
            bad_reviews.append(df[df.review.apply(
                lambda x: any(i in x for i in bad_words) & \
                (df.star_rating <= 3) & (df.labels==label))])
            good_reviews.append(df[df.review.apply(
                lambda x: any(i in x for i in good_words) & \
                (df.star_rating > 3) & (df.labels==label))])

    bad_reviews = pd.concat(bad_reviews).review.drop_duplicates(inplace=True)
    good_reviews = pd.concat(good_reviews).review.drop_duplicates(inplace=True)
    
    return bad_reviews, good_reviews

if __name__ == '__main__':
    files = [file for file in os.listdir('samples/labeled_samples_v3')]
    names = [file.replace('_clustered_embeddings.parquet','') for file in files]
    for file, name in zip(files, names):
        start = time.time()
        try:
            bad, good = get_reviews_having_keywords(file)
            bad.to_parquet(f'{name}_retrieved_bad_reviews.parquet', index=False)
            good.to_parquet(f'{name}_retrieved_good_reviews.parquet', index=False)
            log_and_print(f'Total time: {(time.time()-start)/60:,.2f} minutes')
        except Exception as e:
            logging.info(e.args)
            print('Exception encountered. See log for details. Continuing to next file.')
            continue