import pandas as pd

def optimize_mem_usage(df):
    ''' 
    iterate through all the columns of a dataframe 
    and modify the data type to reduce memory usage.       
    '''
    start_mem = df.memory_usage(deep=True).sum()
    print(f'Memory usage of dataframe is {start_mem:,.0f} bytes')
        
    float_cols = df.select_dtypes('float').columns
    cat_cols = df.select_dtypes('object').columns
    int_cols = df.select_dtypes('int').columns

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