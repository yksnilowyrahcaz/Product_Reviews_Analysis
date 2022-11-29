from bokeh.transform import factor_cmap
from bokeh.plotting import save, output_file, figure
from bokeh.models import ColumnDataSource, HoverTool
from sklearn.feature_extraction.text import TfidfVectorizer
import os, re, time, logging, umap, hdbscan, pandas as pd, colorcet as cc

logging.basicConfig(
    level=logging.INFO,
    filename='log.log',
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s'
    )

def embed_and_cluster(file_path):
    '''
    Embed a data set into two dimensions with UMAP 
    and cluster the embeddings using HDBSCAN.

    param: file_path to data set
    return: None; instead, generates static html file,
            saved to current working directory.
    '''
    msg = f'Processing {file_path} ...'
    logging.info(msg)
    print(msg)

    df = pd.read_parquet(f'data/{file_path}')
    docs = df.product_title.str[:].copy() + ' ' + df.review.str[:].copy()
    docs_sample = docs.sample(min(100000, docs.shape[0]), random_state=1729)
    name = re.findall('(?<=reviews_)[.\w]*(?=.parq)', file_path)[0].lower()    

    print('tfidf vectorizing data ...')
    vectorizer = TfidfVectorizer(min_df=5, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(docs_sample)

    print('learning umap embedding ...')
    embedding = umap.UMAP(
        n_neighbors=30, 
        min_dist=0.0, 
        n_components=2, 
        metric='hellinger'
        ).fit(doc_term_matrix)

    print('learning HDBSCAN clusters ...')
    labels = hdbscan.HDBSCAN(
        min_samples=10,
        min_cluster_size=500
        ).fit_predict(embedding.embedding_)

    print('saving embedding and cluster data ...')
    new_df = pd.concat([
        pd.Series(docs_sample.index).reset_index().rename(columns={0:'_index'}),
        docs_sample.reset_index().rename(columns={0:'review'}),
        pd.DataFrame(embedding.embedding_).rename(columns={0:'c1', 1:'c2'}).reset_index(),
        pd.Series(labels).reset_index().rename(columns={0:'cluster'})
        ], axis=1).drop(['index'], axis=1)
    new_df.cluster = new_df.cluster.astype('category')
    new_df.to_parquet(f'assets/{name}_clustered_embeddings.parquet')

    print('generating plot ...')
    list_x = embedding.embedding_[:,0].tolist()
    list_y = embedding.embedding_[:,1].tolist()
    labels_ = [str(x) for x in labels]
    desc = docs_sample.to_list()

    output_file(
        filename=f'figures/{name}_clustered_embeddings.html', 
        title=f'{" ".join(name.split("_")).title()} Clustered Embeddings')

    source = ColumnDataSource(
        data=dict(x=list_x, y=list_y, desc=desc, clustering=labels_))

    hover = HoverTool(
        tooltips=[
            ('index', '$index'),
            ('(x,y)', '(@x, @y)'),
            ('desc', '@desc')
            ]
        )

    z = figure(width=800, height=800, tools=[hover, 'pan, wheel_zoom, reset'])

    mapper = factor_cmap(
        field_name='clustering', 
        palette=['#EEEEEE']+cc.glasbey[1:len(set(labels_))], 
        factors=list(sorted(set(labels_)))
        )

    z.scatter('x', 'y', source=source, size=1, color=mapper, alpha=0.5)

    save(z)
    msg = f'Processing complete. File saved to figures/{name}_clustered_embeddings.html'
    logging.info(msg)
    print(msg)

if __name__ == '__main__':
    files = [file for file in os.listdir('data') if 'parquet' in file]
    for file in files:
        start = time.time()
        try:
            embed_and_cluster(file)
            print(f'Total time: {(time.time()-start)/60:,.2f} minutes')
        except Exception as e:
            logging.info(e.args)
            print('Exception encountered. See log for details. Continuing to next file.')
            continue  