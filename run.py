import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

########### List of functions


# Use for splitting data for training models

def split_df(df, beta=.7):
    """Split DataFrame on train and test subsets."""
    split = int(df.shape[0] * beta) 
    X = np.array(df.drop(x, axis=1))
    y = df[x]
    X_train, X_valid, y_train, y_valid = X[:split], X[split:], y[:split], y[split:]
    return X_train, X_valid, y_train, y_valid


# Control datasets on object's type

def type_control(df):
    """Delete types "object" from dataset."""
    df = df.select_dtypes(exclude='object')
    return df


# model score

def model_score(model_name):
    model = model_name()
    model.fit(X_train, y_train)
    prediction = model.score(X_valid, y_valid)
    return st.write("{} gives score: ".format(type_of_model), prediction)

###########

data_path = st.sidebar.file_uploader('file', type='csv')
analyse_option = st.sidebar.radio('Choose analysis type', ('Data statistics', 'Features Correlation',
                                                           'Box-plot', 'Distribution of features',
                                                           'Model builder'))

show_all = st.sidebar.checkbox('Show full analysis')
st.title('Web exploratory data analysis')


if data_path is not None:
    df = st.cache(pd.read_csv)(data_path)
    if analyse_option == 'Data statistics' or show_all:
        '''
        ## Data overview
        '''
        st.write(df)
       
        '''
        ## Columns statistics
        '''
        # add choice a column type
        dtype_option = st.selectbox('Choose column\'s type', ( df.dtypes.value_counts().index))
        st.write(df.describe(include=[dtype_option]))

        '''
        ## Ratio of available data (not NAN's) in percent
        '''
        # add values in percent
        data_ratios = (df.count() / len(df))*100
        #st.write(data_ratios * 100)
        st.write(
            f'<div style="float:left;height:200px;width:250px;border:solid 2px #ff0000;"><h2>Левый виджет</h2>{data_ratios}</div>'
            f'<div style="float:left;height:200px;width:300px;border:solid 2px #00ff00;"><h2>Правый виджет</h2></div>',
            unsafe_allow_html=True
        )

    if analyse_option == 'Features Correlation' or show_all:
        '''
        ## Heat map of features
        '''
        corr = df.select_dtypes(exclude=['object', 'datetime']).corr(method='spearman')
        sns.heatmap(corr, annot=True,
                    annot_kws={'size': 8})
        st.pyplot()
        
    if analyse_option=='Box-plot' or show_all:
        '''
        ## Box-plot
        '''
        col_for_box_x = st.selectbox('Choose x axis', [col for col in df.columns if df[col].dtype not in ['object']])
        col_for_box_y = st.selectbox('Choose y axis', [col for col in df.columns if df[col].dtype not in ['object']])
        sns.boxplot(x=col_for_box_x, y=col_for_box_y, data=df)
        st.pyplot()
        
    if analyse_option == 'Distribution of features' or show_all:
        '''
        ## Distribution of features
        '''
        group_cols = st.multiselect('Choose columns for grouping', df.columns)
        anal_col = st.selectbox('Choose column for analysis', [col for col in df.columns if df[col].dtype not in ['object']])
        if len(group_cols) > 0:
            df_gr = df.groupby(group_cols)[anal_col]
        else:
            df_gr = df[anal_col]
        df_gr.hist()
        plt.title('Distribution of {}'.format(anal_col))
        plt.xlabel(anal_col)
        st.pyplot()

    if analyse_option == 'Model builder' or show_all:
        """
        ## Constructor
        """
        st.write("model builder")
        x = st.radio('Choose the Target:', (df.columns))
        st.write('Now target is: ', x)

        # Advice for choosing type of Target
        if df[x].dtypes == 'float':  # or ((x.dtypes == 'int') & ((x.nunique() / df.shape[0]) < .1)):
            "Better choose one of the Regression models"
        else:
            "Better choose one of the Classification models"

        type_of_research = st.selectbox('Choose types of Target:', ('Classification', 'Regression'))


        if type_of_research == "Classification":

            type_of_model = st.selectbox('Choose the model of Classifaer: ', ('KNN', 'Decision Tree', 'Logistic Regression', 'SVM', 'Random Forest'))
            'Now model is: ', type_of_model


            if st.button('calculate'):
                new_df = type_control(df)
                X_train, X_valid, y_train, y_valid = split_df(new_df)

                if type_of_model == 'KNN':
                    from sklearn.neighbors import KNeighborsClassifier
                    model_score(KNeighborsClassifier)

                elif type_of_model == 'Decision Tree':
                    from sklearn.tree import DecisionTreeClassifier
                    model_score(DecisionTreeClassifier)

                elif type_of_model == 'Logistic Regression':
                    from sklearn.linear_model import LogisticRegression
                    model_score(LogisticRegression)

                elif type_of_model == 'SVM':
                    from sklearn.svm import SVC
                    model_score(SVC)

                elif type_of_model == 'Random Forest':
                    from sklearn.ensemble import RandomForestClassifier
                    model_score(RandomForestClassifier)

        elif type_of_research == "Regression":

            type_of_model = st.selectbox("Choose the type of Regressor:", ('Linear Regression', \
                                                                           'Ridge Regression', \
                                                                           'Lasso Regression'))
            if st.button('CALCULATE'):
                new_df = type_control(df)
                X_train, X_valid, y_train, y_valid = split_df(new_df)

                if type_of_model == "Linear Regression":
                    from sklearn.linear_model import LinearRegression
                    model_score(LinearRegression)

                elif type_of_model == "Ridge Regression":
                    from sklearn.linear_model import Ridge
                    model_score(Ridge)

                elif type_of_model == "Lasso Regression":
                    from sklearn.linear_model import Lasso
                    model_score(Lasso)