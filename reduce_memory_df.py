import pandas as pd

def optimize_mem_usage(df, ignore):
    ''' 
    iterate through all the columns of a dataframe 
    and modify the data type to reduce memory usage.

    df: pandas DataFrame to optimize
    ignore: list of columns in df to ignore
    '''
    start_mem = df.memory_usage(deep=True).sum()
    print(f'Memory usage of dataframe is {start_mem:,.0f} bytes')
        
    float_cols = [col for col in df.select_dtypes('float').columns if col not in ignore]
    cat_cols = [col for col in df.select_dtypes('object').columns if col not in ignore]
    int_cols = [col for col in df.select_dtypes('int').columns if col not in ignore]

    for col in df.columns:
        if col in cat_cols:
            df[col] = df[col].astype('category')
        elif col in int_cols:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif col in float_cols:
            df[col] = pd.to_numeric(df[col], downcast='float')

    end_mem = df.memory_usage(deep=True).sum()
    print(f'Memory usage after optimization is: {end_mem:,.0f} bytes')
    print(f'Decreased by {(end_mem - start_mem) / start_mem:.2%}')