import logging, pandas as pd
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] - %(filename)s: line %(lineno)d - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('log.log'), logging.StreamHandler()]
)

def optimize_mem_usage(
    df: pd.DataFrame, 
    ignore: List[str] = []
) -> pd.DataFrame:
    ''' 
    Iterate through all the columns of a pandas DataFrame 
    and downcast or modify types to reduce memory usage.

    :param df: pandas DataFrame to optimize
    
    :param ignore: names of columns in df to ignore
    :type ignore: list of strings
    :default ignore: None
    '''
    start_mem = df.memory_usage(deep=True).sum()
    logging.info(f'Estimated memory usage of dataframe: {start_mem:,.0f} bytes')
        
    float_cols = [col for col in df.select_dtypes('float').columns if col not in ignore]
    cat_cols = [col for col in df.select_dtypes('object').columns if col not in ignore]
    int_cols = [col for col in df.select_dtypes('int').columns if col not in ignore]

    for col in df.columns:
        if col in cat_cols:
            temp_col = df[col].astype('category')
            df[col] = temp_col if temp_col.memory_usage(deep=True) \
                < df[col].memory_usage(deep=True) else df[col]
        elif col in int_cols:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif col in float_cols:
            df[col] = pd.to_numeric(df[col], downcast='float')

    end_mem = df.memory_usage(deep=True).sum()
    logging.info(f'Estimated memory usage after optimization: {end_mem:,.0f} bytes')
    logging.info(f'Estimated percentage decrease: {(end_mem - start_mem) / start_mem:.2%}')