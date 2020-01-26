import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_path = st.sidebar.file_uploader('Choose a csv file', type='csv')
st.title('Exploratory data analysis')

if data_path is not None:
    df = st.cache(pd.read_csv)(data_path)
    '''
    ## Data overview
    '''
    st.write(df)
    '''
    ## Columns statistics
    '''
    st.write(df.describe())
    '''
    ## Ratio of available data (not NAN's)
    '''
    data_ratios = df.count() / len(df)
    st.write(data_ratios)

    #fig, ax = plt.subplots(figsize=(25, 25))
    '''
    ## Heat map of features
    '''
    corr = df.select_dtypes(exclude=['object', 'datetime']).corr(method='spearman')
    sns.heatmap(corr, annot=True,
                annot_kws={'size': 6})
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





