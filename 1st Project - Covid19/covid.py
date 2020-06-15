import math

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA

DATASET_PATH = 'dataset.xlsx'


# Exploration of DATA
def load_dataset(dataset_path=DATASET_PATH):
    """
    Load the dataset and work on a copy
    :param dataset_path: Global variable to define the path where the dataset is located
    :return: Return the copy of the dataset loaded
    """
    data = pd.read_excel(dataset_path)
    df = data.copy()
    return df


def general_info(df, nb_column):
    """
    Display quickly some inforrmation to get started with the analysis of a dataframe
    :param df: Dataframe used
    """
    displayHead(df, 111, True, True, nb_column)
    shapeOfDF(df)
    typeOfDFValues(df)


def displayHead(df, nb_column, every_column=False, every_row=False):
    """
    Display the head of the dataframe df - 5 rows by default
    Using the boolean for every_column and every_row, it's possible to display more
    :param nb_column: Number of column to visualize
    :param df: Dataframe used
    :param every_column: Default is False, if True every column is displayed
    :param every_row: Default is False, if True every row is displayed
    :return:
    """
    if every_column:
        pd.set_option('display.max_column', nb_column)

    if every_row:
        pd.set_option('display.max_row', nb_column)

    print(df.head())
    return df.head()


def shapeOfDF(df):
    """
    Give the shape of the dataframe
    :param df: Dataframe used
    :return: (row_number, column_number)
    """
    print("Shape is : {}".format(df.shape))
    return df.shape


def typeOfDFValues(df):
    """
    Print the types of the values contained in the dataframe
    :param df: Dataframe used
    :return: Return a Series containing counts of unique values
    """
    print(df.dtypes.value_counts())
    return df.dtypes.value_counts()


def checkNan(df):
    """
    Detect missing values.
    :param df: Dataframe used
    :return: Return a boolean same-sized object indicating if the values are NA. NA values, such as None or numpy.NaN,
     gets mapped to True values. Everything else gets mapped to False values
     Characters such as empty strings '' or numpy.inf are not considered NA values
    """
    print(df.isna())
    return df.isna()


def constructHeatMap(data, show=False):
    """
    Build a heatmap of the variables
    :param data: Variables that will build the heatmap
    :param show: Display the plot (Default: False)
    :return:
    """
    plt.figure(figsize=(20, 10))
    sns.heatmap(data, cbar=False)
    if show:
        plt.show()


def missing_values_percentage(df, rate_checked):
    """
    Print out the percentage of the missing values, compared to the rate checked
    :param rate_checked: How many values are missing compared to this rate
    :param df: Dataframe used
    :return: The global percentage of missing values in the dataset
    """
    missing_values = (checkNan(df).sum() / df.shape[0]).sort_values(ascending=True)
    print(len(missing_values[missing_values > rate_checked])  # Ex : rate_checked = 0.98 : 98% of missing values
        / len(missing_values[missing_values > 0.0]))  # Give the percentage of missing values > 90% compared to all
    # the missing values : 68 % (more than half the variables are > 90% of NaN)
    return missing_values


def missing_rate(df):
    """
    Get for each column (feature) the percentage of missing value
    :param df: Dataframe used
    :return: Percentage of missing value
    """
    return df.isna().sum() / df.shape[0]


def keep_values(df, percentage_to_keep=0.9):
    """
    Keeps the values where there are less than a certain percentage of missing values
    :param df: Dataframe used
    :param percentage_to_keep: Percentage to conserve
    :return: A new dataframe where the variables with more than percentage_to_keep are conserved
    """
    return df[df.columns[df.isna().sum() / df.shape[0] < percentage_to_keep]]  # Keep the values where there are
    # less than 90% of missing values


def dropColumn(df, columnName):
    """
    Remove a column
    :param df: Dataframe used
    :param columnName: Name of the column to remove
    :return: A dataframe without the column droped
    """
    return df.drop(columnName, axis=1)


def analyse_target(df, target, normalized=False):
    """
    Compares the number (or percentage) of each value taken by the feature
    :param df: Dataframe used
    :param target: The feature analyzed
    :param normalized: True: Give the proportion of each value taken by the feature
    :return: The number/proportion of each value taken by the feature (ex :
    negative    1
    positive    9

    or

    negative    0.1
    positive    0.9
    """
    print(df[target].value_counts(normalize=normalized))
    return df[target].value_counts(normalize=normalized)


def draw_histograms(df, data_type='float', nb_columns=4):
    """
    Draw the histograms/plot pie of the quantitatives/qualitatives variables
    :param df: Dataframe used
    :param data_type: type of the data : int, int64, float, float64 or object
    :param nb_columns: Number of column for the subplot created to display the plots
    """
    cols = df.select_dtypes(data_type)
    ceiling = math.ceil(len(cols.columns) / nb_columns)
    f, axes = plt.subplots(nb_columns, ceiling, figsize=(7, 7), sharex=True)
    for index, col in enumerate(cols):
        col_index = index % nb_columns
        row_index = index // nb_columns

        if data_type == 'float' or data_type == 'int':
            sns.distplot(df[col], ax=axes[row_index, col_index])

        if data_type == 'object':
            df[col].value_counts().plot.pie()
            plt.show()
    plt.show()


def description_object(df, target):
    """
     Function used on a variable of interest to get the unique values of the column
     For example, let us say we want to find the unique values of column 'continent'
     in a data frame. This would result in all continents in the dataframe.
    :param df: Dataframe used
    :param target: Target we want to find unique()
    :return: numpy.ndarray or ExtensionArray
    """
    return df[target].unique()


def qual_to_quan(df, target, criteria):
    """
    Creates a subset (or collection) of the target with a certain criteria
    Ex: positive_df = qual_to_quan(df, "SARS-Cov-2 exam result", 'positive') creates a subset of the positive results
    to the Covid19 exam
    :param df: Dataframe used
    :param target: Target we want to create a subset from
    :param criteria: Criteria used to create a subset
    :return: A dataframe responding to the criteria chosed
    """
    return df[df[target] == criteria]


def rate_borned(df, missing_rate, rate_inf, rate_sup):
    """
    Creates a subset based on the missing rates
    Ex:
    blood_columns = rate_borned(df, MR, 0.88, 0.9) create column where the missing rate MR is included between
    0.88 and 0.9
    :param df: Dataframe used
    :param missing_rate: missing_rate to compare the rates
    :param rate_inf: Decision rate inf
    :param rate_sup: Decision rate sups
    :return: The column labels of the DataFrame corresponding to the criteria missing rate, rate inf and sup

    """
    return df.columns[(missing_rate < rate_sup) & (missing_rate > rate_inf)]


def display_relations(column_name, relation):
    """
    Display the relation between diff
    display_relations(blood_columns, relation)
    :param column_name: Column the relation are being tested with
    :param relation: List of relation to observe
    Ex : relation = [(positive_df, 'positive'), (negative_df, 'negative')] shows the relation between the
    blood_column and the positive and negative results
    :return:
    """
    for col in column_name:
        plt.figure()
        for rel, lab in relation:
            sns.distplot(rel[col], label=lab)
        plt.legend()
    plt.show()


def count_histogram(df, x, hue, show=True):
    """
    Shows the counts of observations in each categorical bin using bars
    :param show: True to display the plot
    :param df: Dataframe used
    :param x: abscisse
    :param hue:Legend title
    """
    sns.countplot(x=x, hue=hue, data=df)
    if show:
        plt.show()


def crossTable(df, cross1, cross2):
    """
    Compute a simple cross tabulation of two (or more) factors
    :param df: Dataframe used
    :param cross1: First variable to cross with
    :param cross2: Second variable to cross with
    :return: Cross tabulation of the data
    """
    return pd.crosstab(df[cross1], df[cross2])


def crossTables(df, column_name, cross):
    """
    Compute a cross tab for every value of the column with a parameter
    :param df: Dataframe used
    :param column_name: The column where the values are taken from
    :param cross: The parameter which the one the values are crossed with
    """
    cols = column_name.unique()
    ceiling = math.ceil(len(cols) / 5)
    f, axes = plt.subplots(5, ceiling, figsize=(12, 12), sharex=True)
    for index, col in enumerate(cols):
        col_index = index % 5
        row_index = index // 5
        sns.heatmap(pd.crosstab(df[cross], df[col]), annot=True, fmt='d', ax=axes[col_index, row_index])

    plt.show()


def pairwise_relationships(df, variable, cluster=True):
    """
    Display a pairplot, clustermap or heatmap
    :param df: Dataframe used
    :param variable: Variable studied
    :param cluster: True: Clustermap display, False: Heatmap displayed
    """
    sns.pairplot(df[variable])
    if cluster:
        sns.clustermap(df[variable].corr())
    else:
        sns.heatmap(df[variable].corr())
    plt.show()


def view_regression(df, column_name, absc, discrimination):
    """
    Plot data and regression model fits across a FacetGrid
    Ex:
    view_regression(df, blood_columns, "Patient age quantile", "SARS-Cov-2 exam result")
    :param df: Dataframe used
    :param column_name: Columns
    :param absc: Abscisse
    :param discrimination: Discrimination parameter
    """
    for col in column_name:
        plt.figure()
        sns.lmplot(x=absc, y=col, hue=discrimination, data=df)
    plt.show()


def check_correlation(df, value_for_correlation):
    """
    Check if the featurs are correlated
    :param df: Dataframe used
    :param value_for_correlation: Specified the value with which every other parameter would be correlated checked
    :return: The values corresponding to the correation between value_for_correlation and every other parameters
    """
    return df.corr()[value_for_correlation].sort_values()


def relation_in_newcol(df, column, newcol, show=False):
    """
    Display the relation between a specified column and another one
    :param show: True: Display the plot(s)
    :param df: Dataframe used
    :param column: First column
    :param newcol: Second column
    :return:
    """
    for col in column:
        plt.figure()
        for cat in newcol.unique():
            sns.distplot(df[newcol == cat][col], label=cat)
        plt.legend()

    if show:
        plt.show()


def t_test(col, alpha, a, b):
    """
    Calculates the T-test for the means of two independent samples of scores
    This is a two-sided test for the null hypothesis that 2 independent samples have identical average (expected) values
    This test assumes that the populations have identical variances by default
    :param col:
    :param alpha:
    :param a:
    :param b:
    :return:
    """
    stat, p = ttest_ind(a[col].dropna(), b[col].dropna())
    if p < alpha:
        return 'H0 rejected'
    else:
        return 'X'


def student_test(column, t_test):
    """
    Print for each column, the result of the Student's test
    :param column: Column
    :param t_test: Student test
    """
    for col in column:
        print(f'{col :-<50} {t_test(col)}')

    # Exploration of DATA #


def encoding(df, type_values, swap_these_values):
    """
    Encode qualitatives values
    :param df: Dataframe used
    :param type_values: Type of the value worked with
    :param swap_these_values: Swap the values {"Value 0": 0, "Value1": 1, ...}
    :return: A dataframe with the qualitatives values swaped to quantitatives values
    """
    for col in df.select_dtypes(type_values).columns:
        df.loc[:, col] = df[col].map(swap_these_values)
    return df


def is_sick(df, column, parameter):
    return np.sum(df[column[:-2]] == parameter, axis=1) >= 1


def hospitalisation(df):
    if df['Patient addmited to regular ward (1=yes, 0=no)'] == 1:
        return 'surveillance'

    elif df['Patient addmited to semi-intensive unit (1=yes, 0=no)'] == 1:
        return 'semi-intensives'

    elif df['Patient addmited to intensive care unit (1=yes, 0=no)'] == 1:
        return 'ICU'

    else:
        return 'unknown'


def feature_engineering(df, column):
    """
    Feature engineering on column
    :param df: Dataframe used
    :param column: Column targeted with the feature engineering
    :return: A dataframe fetaure engineered
    """
    # df['is sick'] = 'no'
    # df['is sick'] = df[column].sum(axis=1) >= 1
    # df = df.drop(column, axis=1)
    return df


def imputation(df):
    """
    Imputation function
    :param df: Dataframe used
    :return: The dataframe with the imputation applied
    """
    # df['is na'] = (df['Parainfluenza 3'].isna()) | (df['Leukocytes'].isna())
    # df = df.fillna(-999) # not working after few trials
    df = df.dropna(axis=0)
    return df


def preprocessing(df, Target, type_values, swap_these_values, new_feature, column):
    """
    Global preprocessing function
    :param df: Dataframe used
    :param Target: Target variable wanted predicted
    :param type_values: Type of the values
    :param swap_these_values: Values swaped
    :param new_feature: New_feature used for the feature engineering
    :param column: Column concerned by the feature engineering
    :return: X: Features and y: Target to predict
    """
    df = encoding(df, type_values, swap_these_values)
    feature_engineering(df, column)
    df = imputation(df)
    X = dropColumn(df, Target)
    y = df[Target]
    return X, y


def evaluation(model, X_train, y_train, X_test, y_test):
    """
    Evaluation of the model, compare on the same plot the prediction on the trainset and the testset
    Display the confusion matrix and classification report
    Shows the overfitting/undefitting
    :param model: model used
    :param X_train: X trainset
    :param y_train: y trainset
    :param X_test: X testset
    :param y_test: y testset
    """
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)

    print(confusion_matrix(y_test, ypred))
    print(classification_report(y_test, ypred))

    N, train_score, val_score = learning_curve(model, X_train, y_train, cv=4, scoring='f1',
                                               train_sizes=np.linspace(0.1, 1, 10))

    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label='Train score')
    plt.plot(N, val_score.mean(axis=1), label='Validation score')
    plt.legend()


def exploration_of_data():
    df = load_dataset(dataset_path=DATASET_PATH)
    # # general_info(df, 111)
    # NaN = checkNan(df)
    # constructHeatMap(NaN)
    # print(missing_values_percentage(df, 0.98))
    # print(missing_rate(df))

    df = keep_values(df, percentage_to_keep=0.9)
    df = dropColumn(df, 'Patient ID')
    # analyse_target(df, "SARS-Cov-2 exam result", True)
    #
    # draw_histograms(df, 'object')


    """
    Target/Variables relation :
    """
    # Positive and Negative collections
    positive_df = qual_to_quan(df, "SARS-Cov-2 exam result", 'positive')
    negative_df = qual_to_quan(df, "SARS-Cov-2 exam result", 'negative')

    # Blood and Viral collections
    MR = missing_rate(df)
    blood_columns = rate_borned(df, MR, 0.88, 0.9)
    viral_columns = rate_borned(df, MR, 0.75, 0.88)

    relation = [(positive_df, 'positive'), (negative_df, 'negative')]
    # display_relations(blood_columns, relation)


    # Relation Target and Age
    # count_histogram(df, 'Patient age quantile', 'SARS-Cov-2 exam result')

    # Relation, comparison between collection : Target and Viral
    # print(crossTable(df, 'SARS-Cov-2 exam result', 'Influenza A'))
    # crossTables(df, viral_columns, "SARS-Cov-2 exam result")

    """
    Advanced analysis
    """

    # Blood / Blood relations
    pairwise_relationships(df, blood_columns)

    # Blood / Age relations view_regression(df, blood_columns, "Patient age quantile", "SARS-Cov-2 exam result")  # A
    # bit messy, using correslation instead, could be used for more analysis print(check_correlation(df, "Patient age
    # quantile"))  # Check if age is correlated with anything ?

    # Blood / Age relations
    # print(crossTable(df, 'Influenza A', 'Influenza A, rapid test'))
    # print(crossTable(df, 'Influenza B', 'Influenza B, rapid test'))

    # Viral / Blood relations
    df['is sick'] = is_sick(df, viral_columns, 'detected')

    # Relation Sickness / Blood_data
    sick_df = qual_to_quan(df, "is sick", True)
    not_sick_df = qual_to_quan(df, "is sick", False)

    relation = [(sick_df, 'is sick'), (not_sick_df, 'is not sick')]
    # display_relations(blood_columns, relation)

    # Relation Hospitalisation / is Sick
    df['status'] = df.apply(hospitalisation, axis=1)
    # print(df.head())

    # Relation Hospitalisation / Blood
    relation_in_newcol(df, blood_columns, df['status'])

    # Student's Test : needs to have balanced sample
    positive_df.shape  # (558, 38)
    negative_df.shape  # (5086, 38)

    balanced_neg = negative_df.sample(positive_df.shape[0])  # Same sample dimension

    # for col in blood_columns:
    #  print(f'{col :-<50} {t_test(col, 0.02, balanced_neg, positive_df)}')


if __name__ == "__main__":
    exploration_of_data()

    # Preprocessing #
"""
    df2 = load_dataset(dataset_path=DATASET_PATH)
    MR2 = missing_rate(df2)
    blood_columns2 = list(rate_borned(df2, MR2, 0.88, 0.9))
    viral_columns2 = list(rate_borned(df2, MR2, 0.75, 0.88))
    important_columns = ['Patient age quantile', 'SARS-Cov-2 exam result']
    df2 = df2[important_columns + blood_columns2]  # + viral_columns2]
    # print(displayHead(df2, 111, every_column=True, every_row=False))
    trainset, testset = train_test_split(df2, test_size=0.2, random_state=0)

    # Encoding
    swap_values = {'positive': 1, 'negative': 0, 'detected': 1, 'not_detected': 0}
    target = 'SARS-Cov-2 exam result'
    X_train, y_train = preprocessing(trainset, target, 'object', swap_values, 'is sick',
                                     viral_columns2)
    X_test, y_test = preprocessing(testset, target, 'object', swap_values, 'is sick',
                                   viral_columns2)

    # Modelisation
    # model = make_pipeline(SelectKBest(f_classif, k=7), RandomForestClassifier(random_state=0))

    # Machine Learning models
    preprocessor = make_pipeline(PolynomialFeatures(2, include_bias=False), SelectKBest(f_classif, k=10))
    RandomForest = make_pipeline(preprocessor, RandomForestClassifier(random_state=0))
    AdaBoost = make_pipeline(preprocessor, AdaBoostClassifier(random_state=0))
    Svm = make_pipeline(preprocessor, StandardScaler(),SVC(random_state=0))
    KNN = make_pipeline(preprocessor, StandardScaler(), KNeighborsClassifier())
    list_of_models = [RandomForest, AdaBoost, Svm, KNN]
    # Eval Procedure
    for model in list_of_models:
        evaluation(model, X_train, y_train, X_test, y_test)

    # pd.DataFrame(model, index=X_train.columns).plot.bar()  # useful to check what variables are
    # rly important


"""
