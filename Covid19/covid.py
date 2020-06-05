import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve

DATASET_PATH = 'dataset.xlsx'


# Exploration of DATA

def load_dataset(dataset_path=DATASET_PATH):
    data = pd.read_excel(dataset_path)
    df = data.copy()
    return df


def general_info(df):
    displayHead(df, True, True)
    shapeOfDF(df)
    typeOfDFValues(df)


def displayHead(df, every_column=False, every_row=False):
    if every_column:
        pd.set_option('display.max_column', 111)

    if every_row:
        pd.set_option('display.max_row', 111)

    # print(df.head())
    return df.head()


def shapeOfDF(df):
    print("Shape is : {}".format(df.shape))
    return df.shape


def typeOfDFValues(df):
    print(df.dtypes.value_counts())
    return df.dtypes.value_counts()


def checkNan(df):
    # print(df.isna())
    return df.isna()


def constructHeatMap(data, show=False):
    plt.figure(figsize=(20, 10))
    sns.heatmap(data, cbar=False)
    if show:
        plt.show()


def missing_values_percentage(df):
    missing_values = (checkNan(df).sum() / df.shape[0]).sort_values(ascending=True)
    # print(len(missing_values[missing_values > 0.9])  # Ex : 0.98 = 98% of missing values
    #     / len(missing_values[missing_values > 0.0]))  # Give the percentage of missing values > 90% compared to all
    # the missing values : 68 % (more than half the variables are > 90% of NaN)
    return missing_values


def missing_rate(df):
    return df.isna().sum() / df.shape[0]


def keep_values(df, percentage_to_keep=0.9):
    return df[df.columns[df.isna().sum() / df.shape[0] < percentage_to_keep]]  # Keep the values where there are
    # less than 90% of missing values


def dropColumn(df, colonName):
    return df.drop(colonName, axis=1)


def analyse_target(df, target, normalized=False):
    return df[target].value_counts(normalize=normalized)


def draw_histograms(df, data_type='float'):
    for col in df.select_dtypes(data_type):
        if data_type == 'float' or data_type == 'int':
            sns.distplot(df[col])

        if data_type == 'object':
            plt.figure()
            df[col].value_counts().plot.pie()

        plt.show()


def description_object(df, target):
    return df[target].unique()


def qual_to_quan(df, target, criteria1):
    return df[df[target] == criteria1]


def rate_borned(df, missing_rate, rate_inf, rate_sup):
    return df.columns[(missing_rate < rate_sup) & (missing_rate > rate_inf)]


def display_relations(column_name, relation):
    for col in column_name:
        plt.figure()
        for rel, lab in relation:
            sns.distplot(rel[col], label=lab)
        plt.legend()
    plt.show()


def count_histogram(df, x, hue):
    sns.countplot(x=x, hue=hue, data=df)
    plt.show()


def crossTable(df, cross1, cross2):
    return pd.crosstab(df[cross1], df[cross2])


def crossTables(df, column_name, cross):
    for col in column_name:
        plt.figure()
        sns.heatmap(pd.crosstab(df[cross], df[col]), annot=True, fmt='d')
    plt.show()


def pairwise_relationships(df, variable, cluster=True):
    # sns.pairplot(df[variable])
    if cluster:
        sns.clustermap(df[variable].corr())
    else:
        sns.heatmap(df[variable].corr())
    plt.show()


def view_regression(df, column_name, absc, discrimination):
    for col in column_name:
        plt.figure()
        sns.lmplot(x=absc, y=col, hue=discrimination, data=df)
    plt.show()


def check_correlation(df, value_for_correlation):
    return df.corr()[value_for_correlation].sort_values()


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


def relation_in_newcol(df, column, newcol):
    for col in column:
        plt.figure()
        for cat in newcol.unique():
            sns.distplot(df[newcol == cat][col], label=cat)
        plt.legend()

    plt.show()


def t_test(col, alpha, a, b):
    stat, p = ttest_ind(a[col].dropna(), b[col].dropna())
    if p < alpha:
        return 'H0 rejetée'
    else:
        return 0


def student_test(column, t_test):
    for col in column:
        print(f'{col :-<50} {t_test(col)}')

    # Exploration of DATA #


def encoding(df, type_values, swap_these_values):
    for col in type_values:
        df.loc[:, col] = df[col].map(swap_these_values)
    return df


def imputation(df):
    return df.dropna(axis=0)


def preprocessing(df, target, type_values, swap_these_values):
    df = encoding(df, type_values, swap_these_values)
    df = imputation(df)

    X = df.drop(target, axis=1)
    y = df[target]

    print(y.value_counts())
    return X, y


def evaluation(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)

    print(confusion_matrix(y_test, ypred))
    print(classification_report(y_test, ypred))


def exploration_of_data():
    df = load_dataset(dataset_path=DATASET_PATH)
    # general_info(df)
    displayHead(df, True, True)
    NaN = checkNan(df)
    # constructHeatMap(NaN)
    missing_values_percentage(df)
    df = keep_values(df, percentage_to_keep=0.9)
    df = dropColumn(df, 'Patient ID')
    # analyse_target(df, "SARS-Cov-2 exam result", True)
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
    Advanced anlysis
    """

    # Blood / Blood relations
    # pairwise_relationships(df, blood_columns)

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
    # Preprocessing #

    df2 = load_dataset(dataset_path=DATASET_PATH)
    MR2 = missing_rate(df2)
    blood_columns2 = list(rate_borned(df2, MR2, 0.88, 0.9))
    viral_columns2 = list(rate_borned(df2, MR2, 0.75, 0.88))
    important_columns = ['Patient age quantile', 'SARS-Cov-2 exam result']
    df2 = df2[important_columns + blood_columns2 + viral_columns2]
    # print(displayHead(df2, every_column=True, every_row=False))
    trainset, testset = train_test_split(df2, test_size=0.2, random_state=0)

    # Encoding
    swap_values = {'positive': 1, 'negative': 0, 'detected': 1, 'not_detected': 0}
    target = 'SARS-Cov-2 exam result'
    X_train, y_train = preprocessing(trainset, target, df2.select_dtypes('object').columns, swap_values)
    X_test, y_test = preprocessing(testset, target, df2.select_dtypes('object').columns, swap_values)

    # Modelisation
    model = DecisionTreeClassifier(random_state=0)

    # Eval Procedure
    evaluation(model, X_train, y_train, X_test, y_test)

