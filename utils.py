from bokeh.transform import factor_cmap
import yake, pandas as pd, colorcet as cc
from bokeh.plotting import save, output_file, figure
from bokeh.models import ColumnDataSource, HoverTool, Legend
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

def plot_embeddings(df, name):
    title = name.title().replace("_"," ")    
    desc = df.docs.to_list()

    output_file(filename=f'samples/{name}_embeddings.html', title=f'{title} Embeddings')

    source = ColumnDataSource(data=dict(
        x=df.e1.tolist(), y=df.e2.tolist(), desc=desc))

    hover = HoverTool(
        tooltips=[('index', '$index'),('(x,y)', '(@x, @y)'),('desc', '@desc')])

    f = figure(width=800, height=800, tools=[hover, 'pan, wheel_zoom, reset'])
    f.title.text = f'UMAP Applied to a Random Sample of {df.shape[0]:,} Reviews'
    f.title.text_font_size = '15px'
    f.title.align = 'center'

    f.scatter('x', 'y', source=source, size=1, color='navy', alpha=0.7)

    save(f)

def plot_clustered_embeddings(df, name):
    labels = df.labels.tolist()
    title = name.title().replace("_"," ")
    desc = df.docs.to_list()

    output_file(filename=f'samples/{name}_clusters.html', title=f'{title} Clusters')

    source = ColumnDataSource(
        data=dict(x=df.e1.tolist(), y=df.e2.tolist(), desc=desc, labels=labels))

    hover = HoverTool(
        tooltips=[('index', '$index'), ('(x,y)', '(@x, @y)'), ('desc', '@desc')])

    f = figure(width=935, height=800, tools=[hover, 'pan, wheel_zoom, reset'])
    f.title.text = 'HDBSCAN Applied to UMAP Embeddings. Clusters Labeled Using YAKE'
    f.title.text_font_size = '15px'
    f.title.align = 'center'

    labels = [x for x in labels if x != 'unclustered']
    mapper = factor_cmap(
        field_name='labels',
        palette=['#EEEEEE']+cc.glasbey[:len(set(labels))], 
        factors=['unclustered']+list(set(labels))
        )

    f.add_layout(Legend(),'right')
    f.scatter('x', 'y', legend_group='labels', source=source, size=1, color=mapper, alpha=0.7)

    save(f)

def keywords(text):
    kwx = yake.KeywordExtractor(n=1, top=30)
    return ' '.join([x[0] for x in kwx.extract_keywords(text)])  

def get_lex_fields(df, name):
    bad = df[df.star_rating <= 3].labels.value_counts()
    good = df[df.star_rating > 3].labels.value_counts()
    dff = pd.concat([bad.sort_index(), good.sort_index()], axis=1).reset_index()
    dff.columns = 'labels','bad','good'
    dff['ranking'] = dff.bad/dff.good
    dff['product_category'] = name.replace('_',' ')

    stop_words = ENGLISH_STOP_WORDS.union({'star','stars'})
    bad_vecs = CountVectorizer(min_df=5, stop_words=stop_words)
    good_vecs = CountVectorizer(min_df=5, stop_words=stop_words)

    x = {}
    for g in df.groupby('labels'):
        bad_set = set(bad_vecs.fit(g[1][g[1].star_rating <= 3].review).vocabulary_.keys())
        good_set = set(good_vecs.fit(g[1][g[1].star_rating > 3].review).vocabulary_.keys())
        x[g[0]] = {
            'bad_set':' '.join(bad_set.difference(good_set)), 
            'good_set':' '.join(good_set.difference(bad_set))}

    df = pd.DataFrame(x).T.applymap(keywords).sort_index().reset_index()

    return pd.concat([dff, df], axis=1).drop(columns='index')