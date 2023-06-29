from pathlib import Path
from bokeh.transform import factor_cmap
import colorcet as cc, pandas as pd, yake
from bokeh.plotting import save, output_file, figure
from bokeh.models import ColumnDataSource, HoverTool, Legend
from sklearn.feature_extraction.text import CountVectorizer

Path(Path.cwd(), 'plots').mkdir(parents=True, exist_ok=True)

def plot_embeddings(df: pd.DataFrame, dataset_name: str) -> None:
    output_file(
        filename=f'plots/embedded_{dataset_name.replace(" ", "_")}.html', 
        title=f'{dataset_name} Embeddings'
    )
    source = ColumnDataSource(
        data=df[['e1', 'e2', 'docs', 'star_rating']].to_dict('list')
    )
    hover = HoverTool(
        tooltips=[('rating', '@star_rating'), ('text', '@docs')]
    )
    f = figure(width=800, height=800, tools=[hover, 'pan, wheel_zoom, reset'])
    f.title.text = f'UMAP Applied to a Random Sample of {df.shape[0]:,} {dataset_name} Reviews.'
    f.title.text_font_size = '15px'
    f.title.align = 'center'
    f.scatter('e1', 'e2', source=source, size=1, color='navy', alpha=0.7)
    save(f)

def plot_clustered_embeddings(df: pd.DataFrame, dataset_name: str) -> None:
    output_file(
        filename=f'plots/clustered_{dataset_name.replace(" ", "_")}.html', 
        title=f'{dataset_name} Clusters'
    )
    source = ColumnDataSource(
        data=df[['e1', 'e2', 'docs', 'star_rating', 'topics']].to_dict('list')
    )
    hover = HoverTool(
        tooltips=[('topic', '@topics'), ('rating', '@star_rating'), ('text', '@docs')]
    )
    f = figure(width=935, height=800, tools=[hover, 'pan, wheel_zoom, reset'])
    f.title.text = f'HDBSCAN Applied to {dataset_name} UMAP Embeddings. Clusters Labeled Using YAKE.'
    f.title.text_font_size = '15px'
    f.title.align = 'center'
    unique_topics = df['topics'].unique()
    clustered_topics = [topic for topic in unique_topics if topic != 'unclustered']
    mapper = factor_cmap(
        field_name='topics',
        palette=['#EEEEEE'] + cc.glasbey[:len(clustered_topics)], 
        factors=['unclustered'] + clustered_topics
    )
    f.add_layout(Legend(),'right')
    f.scatter('e1', 'e2', legend_group='topics', source=source, size=1, color=mapper, alpha=0.7)
    save(f)

def get_keywords(text: str) -> str:
    kwx = yake.KeywordExtractor(n=1, top=30)
    return ' '.join([word[0] for word in kwx.extract_keywords(text)])  

def get_sentiment_keywords(df: pd.DataFrame, dataset_name: str, stop_words: set) -> pd.DataFrame:
    '''
    Extract keywords for each label in bad (1-3 stars) and good (4-5 stars) reviews
    '''
    bad_label_counts = df[df['star_rating'] <= 3]['topics'].value_counts().sort_index()
    good_label_counts = df[df['star_rating'] > 3]['topics'].value_counts().sort_index()
    rating_df = pd.concat([bad_label_counts, good_label_counts], axis=1).reset_index()
    rating_df.columns = 'topics', 'bad', 'good'
    rating_df['ranking'] = rating_df['bad'] / rating_df['good']
    rating_df['product_category'] = dataset_name.lower()
    rating_df['product_topic'] = rating_df['product_category'] + ' ' + rating_df['topics']
    bad_vectorizer = CountVectorizer(min_df=5, stop_words=stop_words)
    good_vectorizer = CountVectorizer(min_df=5, stop_words=stop_words)
    
    lex_fields = {}
    for g in df.groupby('topics'):
        bad_set = set(
            bad_vectorizer.fit(
                g[1][g[1]['star_rating'] <= 3]['review']
            ).vocabulary_.keys()
        )
        good_set = set(
            good_vectorizer.fit(
                g[1][g[1]['star_rating'] > 3]['review']
            ).vocabulary_.keys()
        )
        lex_fields[g[0]] = {
            'bad_set': ' '.join(bad_set.difference(good_set)), 
            'good_set': ' '.join(good_set.difference(bad_set))
        }
    lex_fields_df = pd.DataFrame(lex_fields).T.applymap(get_keywords).sort_index().reset_index()
    return pd.concat([rating_df, lex_fields_df], axis=1).drop(columns='index')