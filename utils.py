import colorcet as cc
from bokeh.transform import factor_cmap
from bokeh.plotting import save, output_file, figure
from bokeh.models import ColumnDataSource, HoverTool, Legend

def plot_embeddings(df, name):
    title = name.title().replace("_"," ")    
    descs = df.docs.to_list()

    output_file(filename=f'samples/{name}_embeddings.html', title=f'{title} Embeddings')

    source = ColumnDataSource(data=dict(
        x=df.e1.tolist(), y=df.e2.tolist(), descs=descs))

    hover = HoverTool(
        tooltips=[('index', '$index'),('(x,y)', '(@x, @y)'),('desc', '@descs')])

    f = figure(width=800, height=800, tools=[hover, 'pan, wheel_zoom, reset'])
    f.title.text = f'UMAP Applied to a Random Sample of {df.shape[0]:,} Reviews'
    f.title.text_font_size = '15px'
    f.title.align = 'center'

    f.scatter('x', 'y', source=source, size=1, color='navy', alpha=0.7)

    save(f)

def plot_clustered_embeddings(df, name):
    _labels = df.labels.tolist()
    title = name.title().replace("_"," ")
    descs = df.docs.to_list()

    output_file(filename=f'samples/{name}_clusters.html', title=f'{title} Clusters')

    source = ColumnDataSource(
        data=dict(x=df.e1.tolist(), y=df.e2.tolist(), desc=descs, labels=_labels))

    hover = HoverTool(
        tooltips=[('index', '$index'),('(x,y)', '(@x, @y)'),('desc', '@descs')])

    f = figure(width=935, height=800, tools=[hover, 'pan, wheel_zoom, reset'])
    f.title.text = 'HDBSCAN Applied to UMAP Embeddings. Clusters Labeled Using YAKE'
    f.title.text_font_size = '15px'
    f.title.align = 'center'

    _labels.remove('unclustered')
    mapper = factor_cmap(
        field_name='labels',
        palette=['#EEEEEE']+cc.glasbey[:len(set(_labels))], 
        factors=['unclustered']+list(set(_labels))
        )

    # f.add_layout(Legend(),'right')
    f.scatter('x', 'y', legend_group='labels', source=source, size=1, color=mapper, alpha=0.7)

    save(f)