# Analyzing Amazon Product Reviews

Using TFIDF, UMAP, and HDBSCAN to analyze product reviews

<p align="center">
<img src="images/review.jpg" width=600>
<p/>

Image: Getty Images

## Table of Contents
1. [File Descriptions](#files)
2. [Supporting Packages](#packages)
3. [How To Use This Repository](#howto)
4. [Project Motivation](#motivation)
5. [About The Dataset](#data)
6. [Resources](#resources)
7. [Acknowledgements](#acknowledgements)
8. [Licence & copyright](#license)

## File Descriptions <a name="files"></a>
| File | Description |
| :--- | :--- |
| code/cluster.py | cluster reviews and find key topics |
| code/logger.py | configure basic logging |
| code/preprocess.py | prep data set and convert from tsv to parquet |
| code/reduce_df_memory.py | downcast pandas data types to conserve memory |
| code/sample.py | sample reviews for each cluster key topic |
| code/utils.py | functions for plotting and finding keywords |
| requirements.in | pip-tools spec file for requirements.txt |
| requirements.txt | list of python dependencies |

## Supporting Packages <a name="packages"></a>
In addition to the standard python library, this analysis utilizes the following packages:
- [bokeh](https://bokeh.org/)
- [colorcet](https://colorcet.holoviz.org/)
- [dask](https://www.dask.org/)
- [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/)
- [pandas](https://pandas.pydata.org/docs/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [UMAP](https://umap-learn.readthedocs.io/en/latest/)
- [YAKE](https://pypi.org/project/yake/)

Please see `requirements.txt` for a complete list of packages and dependencies used in the making of this project.

## How To Use This Repository <a name="howto"></a>
1. Clone the repo locally or download and unzip this repository to a local machine.
2. Navigate to this directory and open the command line. For the purposes of running the scripts, this will be the root directory.
3. Create a virtual environment to store the supporting packages. The `--upgrade-deps` option is availble for python 3.9+.

        python -m venv venv --upgrade-deps

4. Activate the virtual environment.

        venv\scripts\activate (windows) or
        venv/bin/activate (linux)

5. Install the supporting packages from the requirements.txt file. Note, at this time of this writing there is a known issue with pynndescent. See https://github.com/lmcinnes/umap/issues/1032. Thus, we use version 0.5.8, which appears to work.

        pip install -r requirements.txt

6. Download the [Amazon US Customer Reviews Dataset](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset) from Kaggle, create a folder called "data" in the root of this cloned repo and move the downloaded data sets there. This demo utilizes all of the available product category data sets, including the multilingual data set, but the scripts can be run on a single data set. The complete data set is about 22GB, takes about 33 minutes to download the zip file and about 15 minutes to unzip all the tsv files. If you prefer, you can access the data via Amazon S3 bucket in parquet format.

7. Preprocess the tsv files. This command will preprocess all tsv files in the data folder. In my case, this took an average of 1.5 minutes per file and one hour to preprocess all 37 files sequentially on a laptop with four cores (eight processors) and 12GB of RAM. It's recommended to use parallel processing instead. To preprocess specific files, import the `preprocess` function from `preprocess.py` or update the `if __name__ == __main__:` block as needed.

        python code/preprocess.py

8. Cluster the reviews and extract key topics and keywords. This command will apply to all files in the data folder with suffix "*preprocessed.parquet". In my case, this took an average of 8.2 minutes per file and about five hours to preprocess all 37 files sequentially. It's recommended to use parallel processing instead. To cluster specific files, import the `run_pipeline` function from `cluster.py` or update the `if __name__ == __main__:` block as needed.

        python code/cluster.py

Running `cluster.py` will create two files for each data set:
- `*sample_key_topics.parquet`, which is a sample of, at most, 100k reviews, along with their two-dimensional embeddings, cluster assigments, and cluster key topics.
- `*sample_keywords.parquet`, which is a summary of keywords per key topic split between negative (1-3 star rating) and positive (4-5 star rating) reviews.

`combined_keywords.parquet` is also created and combines all `*sample_keywords.parquet` files into one file for use in a dashboarding tool, such as Power BI.

9. Sample reviews by key topic for use in a dashboard. Unlike the previous scripts, `sample.py` only takes about 1-2 minutes to run.

        python code/sample.py

Running `sample.py` will create will create two files: 
- `bad_reviews_sample.parquet`, which includes a sample of, at most, 5k bad (1-3 star rating) reviews from each `*sample_key_topics.parquet` file.
- `good_reviews_sample.parquet`,  which includes a sample of, at most, 5k good (4-5 star rating) reviews from each `*sample_key_topics.parquet` file.

## Project Motivation <a name="motivation"></a>
This project was inspired by the 2022 annual Data Mining Competition organized by RSM. The primary question posed was "What can product reviews tell us?"

## About The Dataset <a name="data"></a>
The [Amazon US Customer Reviews Dataset](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset) from Kaggle consists of millions of customer product reviews that span the period of 1995-2015 and a wide range of product categories, from baby products to software.

## Resources <a name="resources"></a>
[Understanding UMAP](https://pair-code.github.io/understanding-umap/)

## Acknowledgements <a name="acknowledgements"></a>
Thank you to Cynthia Rempel for sharing the data set on Kaggle. Thank you to my Data Mining Competition teammates, Caroline Ingram, Manuj Aggarwal, and Nathan Gossage for their collaboration. We would like to thank RSM for organizing this event.

## License & copyright <a name="license"></a>
© Zachary Wolinsky 2023

Licensed under the [MIT License](LICENSE.txt)
