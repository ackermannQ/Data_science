from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from data_lib import *

DATASET_PATH = 'bank-additional-full.csv'

# Exploration of DATA

if __name__ == "__main__":
    # Preprocessing #
    df = load_dataset(dataset_path=DATASET_PATH, separator=';')
    # general_info(df)
    displayHead(df, True, True)
    # print(missing_values_percentage(df)) # None
    # print(missing_rate(df)) # 0% missing in each category
    # print(analyse_target(df, 'y', normalized=True))
    # draw_histograms(df, data_type='float64')
    # print(description_object(df, 'y'))
    """
    Target/Variables relation :
    """
    # Yes and No subsets
    yes_df = subset_creator(df, 'y', 'yes')
    no_df = subset_creator(df, 'y', 'no')
    # Age, profession, mois -> y ?
    # df = df[['age', 'job', 'month', 'y']]
    df2 = load_dataset(dataset_path=DATASET_PATH, separator=';')
    df2.drop(['job', 'month', 'day_of_week', 'pdays', 'marital', 'default'], axis=1, inplace=True)
    swap_values = {'yes': 1, 'no': 0, 'telephone': 0, 'cellular': 1, 'nonexistent': 0, 'failure': 0, 'success': 1}
    encoding(df2, 'object', swap_values)
    df2.drop(['education'], axis=1, inplace=True)

    trainset, testset = train_test_split(df2, test_size=0.2, random_state=0)
    target = 'y'
    # swap_values = {'yes': 1, 'no': 0, 'mar': 0, 'apr': 1, 'may': 2, 'jun': 3, 'jul': 4, 'aug': 5, 'sep': 6, 'oct': 7,
    #                'nov': 8, 'dec': 9}

    X_train, y_train = preprocessing(trainset, target, 'object', swap_values)
    X_test, y_test = preprocessing(testset, target, 'object', swap_values)

    model = make_pipeline(SelectKBest(f_classif, k=2), RandomForestClassifier(random_state=0))

    random_state = 11
    # LogisticRegression
    LR = Pipeline([('lr', LogisticRegression(random_state=random_state, max_iter=500))])
    # RandomForestClassifier
    RF = Pipeline([('rf', RandomForestClassifier(random_state=random_state, oob_score=True))])
    # KNeighborsClassifier
    KNN = Pipeline([('knn', KNeighborsClassifier())])
    # DecisionTreeClassifier
    DT = Pipeline([('dt', DecisionTreeClassifier(random_state=random_state, max_features='auto'))])
    # BaggingClassifier
    # note we use SGDClassifier as classier inside BaggingClassifier
    BAG = Pipeline([('bag', BaggingClassifier(
        base_estimator=SGDClassifier(random_state=random_state, max_iter=1500),
        random_state=random_state, oob_score=True))])

    list_of_models = [LR, RF, KNN, DT, BAG]

    # # # Eval Procedure
    # for model in list_of_models:
    #     model.fit(X_train, y_train)
    #     ypred = model.predict(X_test)
    #     print(confusion_matrix(y_test, ypred))
    #     print(classification_report(y_test, ypred))
    #
    #     N, train_score, val_score = learning_curve(model, X_train, y_train,
    #                                                cv=4, scoring='f1',
    #                                                train_sizes=np.linspace(0.1, 1, 10))

    # plt.figure(figsize=(12, 8))
    # plt.plot(N, train_score.mean(axis=1), label='train score')
    # plt.plot(N, val_score.mean(axis=1), label='validation score')
    # plt.legend()

    # Model optimization
    cv = StratifiedKFold(shuffle=True, n_splits=5, random_state=random_state)

    # Grid search CV parameters
    # set for LogisticRegression
    grid_params_LR = [{
        'lr__penalty': ['l2'],
        'lr__C': [0.3, 0.6, 0.7]
    }]
    # set for RandomForestClassifier
    grid_params_RF = [{
        'rf__n_estimators': [60, 70]
    }]
    # set for KNeighborsClassifier
    grid_params_KNN = [{'knn__n_neighbors': [16, 17, 18]}]

    # set for DecisionTreeClassifier
    grid_params_DT = [{
        'dt__max_depth': [8, 10],
    }]
    # set for BaggingClassifier
    grid_params_BAG = [{'bag__n_estimators': [10, 15, 20]}]

    gs_LR = GridSearchCV(LR, param_grid=grid_params_LR,
                         scoring='accuracy', cv=cv)
    # for RandomForestClassifier
    gs_RF = GridSearchCV(RF, param_grid=grid_params_RF,
                         scoring='accuracy', cv=cv)
    # for KNeighborsClassifier
    gs_KNN = GridSearchCV(KNN, param_grid=grid_params_KNN,
                          scoring='accuracy', cv=cv)
    # for DecisionTreeClassifier
    gs_DT = GridSearchCV(DT, param_grid=grid_params_DT,
                         scoring='accuracy', cv=cv)
    # for BaggingClassifier
    gs_BAG = GridSearchCV(BAG, param_grid=grid_params_BAG,
                          scoring='accuracy', cv=cv)

    # models that we iterate over
    # dict for later use
    model_dict = {0: 'Logistic_reg', 1: 'RandomForest', 2: 'Knn', 3: 'DesionTree', 4: 'Bagging with SGDClassifier',
                  5: 'SGD Class'}

    result_acc = {}
    result_auc = {}
    models = []
    look_for = [gs_LR, gs_RF, gs_KNN, gs_DT, gs_BAG]

    for index, model in enumerate(look_for):
        print()
        print('+++++++ Start New Model ++++++++++++++++++++++')
        print('Estimator is {}'.format(model_dict[index]))
        model.fit(X_train, y_train)
        print('---------------------------------------------')
        print('best params {}'.format(model.best_params_))
        print('best score is {}'.format(model.best_score_))
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print('---------------------------------------------')
        print('ROC_AUC is {} and accuracy rate is {}'.format(auc, model.score(X_test, y_test)))

        print('++++++++ End Model +++++++++++++++++++++++++++')
        print()
        print()
        models.append(model.best_estimator_)
        result_acc[index] = model.best_score_
        result_auc[index] = auc

    for k, l, m in model_dict.values(), result_acc.values(), result_auc.values():
        plt.plot(k, l, c='r')
        plt.plot(k, m, c='b')
    plt.xlabel('Models')
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy and ROC_AUC')
    plt.title('Result of Grid Search')
    plt.legend(['Accuracy', 'ROC_AUC'])
    plt.show()
    pd.DataFrame(list(zip(model_dict.values(), result_acc.values(), result_auc.values())),
                 columns=['Model', 'Accuracy_rate', 'Roc_auc_rate'])
    


