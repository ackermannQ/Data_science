from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report, roc_auc_score, recall_score
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA

from data_lib import *

DATASET_PATH = 'dataset.xlsx'


def exploration_of_data():
    df = load_dataset(dataset_path=DATASET_PATH)
    general_info(df, 111)
    NaN = checkNan(df)
    constructHeatMap(NaN)
    print(missing_values_percentage(df, 0.98))
    print(missing_rate(df))

    df = keep_values(df, percentage_to_keep=0.9)
    df = dropColumn(df, 'Patient ID')
    analyse_target(df, "SARS-Cov-2 exam result", True)

    draw_histograms(df, 'object')

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
    display_relations(blood_columns, relation)

    # Relation Target and Age
    count_histogram(df, 'Patient age quantile', 'SARS-Cov-2 exam result')

    # Relation, comparison between collection : Target and Viral
    print(crossTable(df, 'SARS-Cov-2 exam result', 'Influenza A'))
    crossTables(df, viral_columns, "SARS-Cov-2 exam result")

    """
    Advanced analysis
    """

    # Blood / Blood relations
    pairwise_relationships(df, blood_columns)

    # Blood / Age relations
    view_regression(df, blood_columns, "Patient age quantile", "SARS-Cov-2 exam result")  # A
    # bit messy, using correslation instead, could be used for more analysis
    check_correlation(df, "Patient age quantile")  # Check if age is correlated with anything ?

    # Blood / Age relations
    print(crossTable(df, 'Influenza A', 'Influenza A, rapid test'))
    print(crossTable(df, 'Influenza B', 'Influenza B, rapid test'))

    # Viral / Blood relations
    df['is sick'] = np.sum(df[viral_columns[:-2]] == 'detected', axis=1) >= 1

    # Relation Sickness / Blood_data
    sick_df = qual_to_quan(df, "is sick", True)
    not_sick_df = qual_to_quan(df, "is sick", False)

    relation = [(sick_df, 'is sick'), (not_sick_df, 'is not sick')]

    display_relations(blood_columns, relation)

    # Relation Hospitalisation / is Sick
    def hospitalisation(df):
        if df['Patient addmited to regular ward (1=yes, 0=no)'] == 1:
            return 'Surveillance'

        elif df['Patient addmited to semi-intensive unit (1=yes, 0=no)'] == 1:
            return 'Semi-intensives'

        elif df['Patient addmited to intensive care unit (1=yes, 0=no)'] == 1:
            return 'ICU'

        else:
            return 'Unknown'

    df['status'] = df.apply(hospitalisation, axis=1)

    # Relation Hospitalisation / Blood
    relation_in_newcol(df, blood_columns, df['status'], False)

    # Student's Test : needs to have balanced sample
    positive_df.shape  # (558, 38)
    negative_df.shape  # (5086, 38)

    balanced_neg = negative_df.sample(positive_df.shape[0])  # Same sample dimension

    student_test(blood_columns, balanced_neg, positive_df, alpha=0.02)


if __name__ == "__main__":
    # exploration_of_data()

    # Preprocessing #
    # First we recreate the subset we need
    df2 = load_dataset(dataset_path=DATASET_PATH)  # Working on a different version of the dataset is a good practice
    MR2 = missing_rate(df2)
    blood_columns2 = list(rate_borned(df2, MR2, 0.88, 0.9))
    viral_columns2 = list(rate_borned(df2, MR2, 0.75, 0.88))
    important_columns = ['Patient age quantile', 'SARS-Cov-2 exam result']

    df2 = df2[important_columns + blood_columns2]  # + viral_columns2]  # finally not a great impact

    trainset, testset = train_test_split(df2, test_size=0.2, random_state=0)

    # Encoding and Preprocessing
    swap_values = {'positive': 1, 'negative': 0, 'detected': 1, 'not_detected': 0}
    target = 'SARS-Cov-2 exam result'
    X_train, y_train = preprocessing(trainset, target, 'object', swap_values, 'is sick',
                                     viral_columns2)
    X_test, y_test = preprocessing(testset, target, 'object', swap_values, 'is sick',
                                   viral_columns2)

    model1 = RandomForestClassifier(random_state=0)
    build_feature_importance(RandomForestClassifier, X_train, y_train)
    evaluation(model1, X_train, y_train, X_test, y_test)

    # Modelisation
    model2 = make_pipeline(SelectKBest(f_classif, k=7), RandomForestClassifier(random_state=0))
    evaluation(model2, X_train, y_train, X_test, y_test)

    # Multiple machine Learning models tests
    preprocessor = make_pipeline(PolynomialFeatures(2, include_bias=False), SelectKBest(f_classif, k=10))
    RandomForest = make_pipeline(preprocessor, RandomForestClassifier(random_state=0))
    AdaBoost = make_pipeline(preprocessor, AdaBoostClassifier(random_state=0))
    SVM = make_pipeline(preprocessor, StandardScaler(), SVC(random_state=0))
    KNN = make_pipeline(preprocessor, StandardScaler(), KNeighborsClassifier())
    list_of_models = {'RandomForest': RandomForest, 'AdaBoost': AdaBoost,
                      'SVM': SVM, 'KNN': KNN}

    # Evaluation Procedure
    for name, model in list_of_models.items():
        print(name)
        evaluation(model, X_train, y_train, X_test, y_test)

    # Optimization of the SVM model
    # Simple
    grid_params = [{
        'svc__gamma': [1e-3, 1e-4],
        'svc__C': [1, 10, 100, 1000]
    }]

    gs = GridSearchCV(SVM, param_grid=grid_params,
                      scoring='recall', cv=4)

    gs.fit(X_train, y_train)
    print(gs.best_params_)
    y_pred = gs.predict(X_test)

    print(classification_report(y_test, y_pred))

    evaluation(gs.best_estimator_, X_train, y_train, X_test, y_test)

    # Advanced
    grid_params = [{
        'svc__gamma': [1e-3, 1e-4],
        'svc__C': [1, 10, 100, 1000],
        'pipeline__polynomialfeatures__degree': [2, 3, 4],
        'pipeline__selectkbest__k': range(48, 58)
    }]

    gs = RandomizedSearchCV(SVM, grid_params,
                            scoring='recall', cv=4, n_iter=300)

    gs.fit(X_train, y_train)
    print(gs.best_params_)
    y_pred = gs.predict(X_test)

    print(classification_report(y_test, y_pred))

    # evaluation(gs.best_estimator_, X_train, y_train, X_test, y_test)
    precision_recall(X_test, y_test, gs)

    y_pred = model_final(gs.best_estimator_, X_test, threshold=-1)
    print(f1_score(y_test, y_pred))
    print(recall_score(y_test, y_pred))
