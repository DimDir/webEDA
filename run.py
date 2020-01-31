import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# clustering algorithms
from sklearn.cluster import KMeans


########### List of functions


# Use for splitting data for training models
@st.cache
def split_df(df, target, beta=.7):
    """Split DataFrame on train and test subsets."""
    split = int(df.shape[0] * beta) 
    X = np.array(df.drop(target, axis=1))
    y = df[target]
    X_train, X_valid, y_train, y_valid = X[:split], X[split:], y[:split], y[split:]
    return X_train, X_valid, y_train, y_valid


# Control datasets on object's type
@st.cache
def type_control(df, target):
    """Delete types "object" from dataset."""
    df_label = df.drop(target, axis=1)
    # transform object features
    for column in df_label.select_dtypes(include='object').columns:
        encoder = LabelEncoder()
        df_label[f"{column}" + "_lb"] = encoder.fit_transform(df_label[column])
    return df_label.select_dtypes(exclude='object').join(df[target])


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
                                                           'Model builder', 'Feature Importance', 'Clustering'))

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

    # Feature Distribution Histogram
    if analyse_option == 'Distribution of features' or show_all:
        '''
        ## Distribution of features
        '''
        group_cols = st.multiselect('Choose columns for grouping', df.columns)
        anal_col = st.selectbox('Choose column for analysis', df.columns) #[col for col in df.columns if df[col].dtype not in ['object']])
        if len(group_cols) > 0:
            df_gr = df.groupby(group_cols)[anal_col]
        else:
            df_gr = df[anal_col]

        df_gr.hist()
        plt.title('Distribution of {}'.format(anal_col))
        plt.xlabel(anal_col)
        st.pyplot()

        if st.button("LOGARITHMIC HISTOGRAMM"):
            df_lg = np.log(df[anal_col])
            df_lg.hist()
            plt.ylabel("log")
            st.pyplot()

    if analyse_option == 'Model builder' or show_all:
        """
        ## Model builder
        """
        model = None
        x = st.selectbox('Choose the Target:', df.columns)
        #x = st.selectbox('Choose the Target:', df.select_dtypes(exclude='object').columns)
        st.write('Now target is: ', x)

        # Advice for choosing type of Target
        if df[x].dtypes == 'float':  # or ((x.dtypes == 'int') & ((x.nunique() / df.shape[0]) < .1)):
            st.write("Better choose one of the Regression models")
        else:
            st.write("Better choose one of the Classification models")

        type_of_research = st.selectbox('Choose type of prediction:', ('Classification', 'Regression'))

        if type_of_research == "Classification":
            type_of_model = st.selectbox('Choose the model of Classifier: ',
                                         ('KNN', 'Decision Tree', 'Logistic Regression', 'SVM', 'Random Forest'))
            st.write('Now model is: ', type_of_model)
            model_cl_dict = {'KNN': KNeighborsClassifier, 'Decision Tree': DecisionTreeClassifier,
                             'Logistic Regression': LogisticRegression, 'SVM': SVC,
                             'Random Forest': RandomForestClassifier}
            model = model_cl_dict.get(type_of_model)

        elif type_of_research == "Regression":

            type_of_model = st.selectbox("Choose the type of Regressor:", ('Linear Regression',
                                                                           'Ridge Regression',
                                                                           'Lasso Regression'))
            model_reg_dict = {'Linear Regression': LinearRegression, 'Ridge Regression': Ridge,
                             'Lasso Regression': Lasso}
            model = model_reg_dict.get(type_of_model)

        if st.button('CALCULATE'):
            new_df = type_control(df, x)
            X_train, X_valid, y_train, y_valid = split_df(new_df, x)

            if model is not None:
                model_score(model)

    # Feature importance analysis
    if analyse_option == 'Feature Importance' or show_all:

        """
        ## Feature importance
        """
        model = None
        target_name = st.selectbox('Choose the Target:', df.select_dtypes(exclude='object').columns)
        type_of_research = st.selectbox('Choose type of prediction:', ('Classification', 'Regression'))
        if type_of_research == 'Classification':
            rf = RandomForestClassifier()
        else:
            rf = RandomForestRegressor()

        new_df = df.select_dtypes(exclude='object')
        target = new_df[target_name]
        X = new_df.drop(target_name, axis=1)
        X_train, X_valid, y_train, y_valid = train_test_split(X, target, test_size=0.8)

        if st.button('Calculate default feature importance'):
            rf.fit(X_train, y_train)
            st.write('Training score: ', rf.score(X_train, y_train), 'Validation score:', rf.score(X_valid, y_valid))
            """
            ### Default feature importance
            """
            feat_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
            feat_importances.nlargest(10).plot(kind='barh')
            st.pyplot()

    if analyse_option == 'Clustering':

        """
        ## Clustering
        """
        x = st.selectbox('Choose the Target:', df.columns)
        n_clusters = st.number_input(label="Choose number of clasters: ", min_value=1)
        st.write("The number of clastets is: ", n_clusters)

        data, target = df.drop(x, axis=1), df[x]
        kmeans = KMeans(n_clusters=int(n_clusters))
        kmeans.fit(data)
        y_kmeans = kmeans.predict(data)
        plt.scatter(x=data.iloc[:, 1], y=data.iloc[:, 2], c=y_kmeans)
        st.pyplot()























