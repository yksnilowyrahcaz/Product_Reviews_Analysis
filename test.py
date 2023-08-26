import pandas as pd
from pathlib import Path
import hdbscan, pandas as pd, time, umap, yake
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from utils import plot_embeddings, plot_clustered_embeddings, get_sentiment_keywords
file_path = Path('data/amazon_reviews_us_Digital_Music_Purchase_v1_00_preprocessed.parquet')
df = pd.read_parquet(file_path)
# df = pd.read_parquet('data/amazon_reviews_us_Gift_Card_v1_00_preprocessed.parquet')
df = df[(df['verified_purchase'] == 'Y') & (df['helpful_votes'] / df['total_votes'] > 0.5)]
sample = df.sample(min(100_000, df.shape[0]), random_state=1729)
sample['docs'] = df['product_title'].str[:] + ' ' + df['review'].str[:]
dataset_name = sample['product_category'].unique()[0].replace('_', ' ').title() if len(sample['product_category'].unique()) == 1 else 'Multilingual' 
stop_words = ENGLISH_STOP_WORDS.union({'star', 'stars', *dataset_name.lower().split()})
vectorizer = TfidfVectorizer(min_df=5, stop_words=stop_words)
doc_term_matrix = vectorizer.fit_transform(sample['docs'])
embedding = umap.UMAP(n_neighbors=30, min_dist=0.0, verbose=True, n_components=2, metric='hellinger').fit(doc_term_matrix)
sample['e1'] = embedding.embedding_[:, 0]
sample['e2'] = embedding.embedding_[:, 1]
clusters = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500).fit_predict(embedding.embedding_)
sample['clusters'] = clusters
kwd = {-1: 'unclustered'}
kwx = yake.KeywordExtractor(n=1, top=1, stopwords=stop_words)
for cluster in sample[sample['clusters'] != -1].groupby('clusters'):
    text = ' '.join(cluster[1]['docs'].tolist())
    kwd[cluster[0]] = kwx.extract_keywords(text)[0][0].lower()
sample['topics'] = sample['clusters'].map(kwd)
plot_embeddings(sample, dataset_name)
plot_clustered_embeddings(sample, dataset_name)
sample.drop(columns='docs', inplace=True)
sample.to_parquet(f'data/{file_path.stem.replace("preprocessed", "sample")}_key_topics.parquet', index=False)
sentiment_df = get_sentiment_keywords(sample, dataset_name, stop_words)
sentiment_df.to_parquet(f'data/{file_path.stem.replace("preprocessed", "sample")}_keywords.parquet', index=False)