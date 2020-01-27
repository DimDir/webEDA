import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_path = st.sidebar.file_uploader('file', type='csv')
st.title('Web exploratory data analysis')

if data_path is not None:
    df = st.cache(pd.read_csv)(data_path)
    
    '''
    ## Data overview
    '''
    
    st.write(df)
    '''
    ## Columns statistics
    '''
    # add choice a column type    
    dtype_option = st.selectbox('Chose column\'s type', ( df.dtypes.value_counts().index))
    st.write(df.describe(include=[dtype_option]))
    
    
    '''
    ## Ratio of available data (not NAN's) in percent
    '''
    # add values in percent
    data_ratios = (df.count() / len(df))
    st.write(data_ratios * 100)  
    
    
    #fig, ax = plt.subplots(figsize=(25, 25))
    '''
    ## Heat map of features
    '''
    corr = df.select_dtypes(exclude=['object', 'datetime']).corr(method='spearman')
    sns.heatmap(corr, annot=True,
                annot_kws={'size': 8})
    st.pyplot()

    '''
    ## Box-plot
    '''
    col_for_box_x = st.selectbox('Choose x asix', df.columns)
    col_for_box_y = st.selectbox('Choose y asix', df.columns)
    sns.boxplot(x=col_for_box_x, y=col_for_box_y, data=df)
    st.pyplot()
    
    '''
    ## Distribution of features
    '''
    group_cols = st.multiselect('Choose columns for grouping', df.columns)
    anal_col = st.selectbox('Choose column for analysis', df.columns)
    if len(group_cols) > 0:
        df_gr = df.groupby(group_cols)[anal_col]
    else:
        df_gr = df[anal_col]
    df_gr.hist()
    plt.title('Distribution of {}'.format(anal_col))
    plt.xlabel(anal_col)
    st.pyplot()
