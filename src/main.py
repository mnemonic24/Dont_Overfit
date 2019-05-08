import pandas as pd
import numpy as np
import setting
import my_path
import multiprocessing
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeRegressor
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import eli5
from eli5.sklearn import PermutationImportance

# data parameter
ID = 'id'
TARGET = 'target'
dtypes = setting.D_TYPES

# directory path
data_dir_path = my_path.DATA_DIR_PATH
submit_dir_path = my_path.SUBMIT_DIR_PATH
figure_dir_path = my_path.FIGURE_DIR_PATH


# dataset: (train, test)
def load_file(dataset):
    usecols = dtypes.keys()
    if dataset == 'test':
        usecols = [col for col in dtypes.keys() if col != TARGET]
    df = pd.read_csv(data_dir_path + f'{dataset}.csv', encoding='utf-8', dtype=dtypes, usecols=usecols)
    return df


# submit file name in [model, score, now time]
def output_submit_csv(pred, score, index, modelname):
    df_test_pred = pd.Series(pred, index=index, name=TARGET)
    str_nowtime = datetime.now().strftime("%Y%m%d%H%M%S")
    df_test_pred.to_csv(submit_dir_path + f'{modelname}_{str_nowtime}_{round(score * 100, 2)}.csv', header=True)


def nn_model(train, valid, test):
    model = Sequential(Dense(100, input_shape=(300, ), activation='relu'))
    model.add(Dense(64))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(optimizer='Adam', loss='mse')
    model.fit(train[features], train[TARGET], verbose=3, batch_size=250, epochs=5)

    valid_pred = model.predict(valid[features], batch_size=250)
    auc_score = roc_auc_score(valid[TARGET], valid_pred)
    print(auc_score)
    print(classification_report(valid[TARGET], valid_pred))

    test_pred = model.predict(test[features])
    return test_pred


# Logistic Regression (plot score and coef)
def lgr_model(train, valid, test):
    params = dict(C=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                  penalty=['l2'],
                  solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                  )
    clf = LogisticRegression(class_weight='balanced', random_state=0)
    cv = GridSearchCV(estimator=clf, param_grid=params, scoring='roc_auc', n_jobs=-1, cv=10, verbose=1)
    cv.fit(train[top_features], train[TARGET])
    print(cv.best_params_)

    valid_pred = cv.predict(valid[top_features])
    auc_score = roc_auc_score(valid[TARGET], valid_pred)
    print(auc_score)
    print(classification_report(valid[TARGET], valid_pred))

    test_pred = cv.predict(test[top_features])

    return test_pred, auc_score


# RandomForest Classification (plot score and return importance features)
def rfc_model(train, valid):
    clf = RandomForestClassifier(n_estimators=1000, class_weight='balanced', max_depth=3, random_state=0)
    clf.fit(train[features], train[TARGET])

    valid_pred = clf.predict(valid[features])
    auc_score = roc_auc_score(valid[TARGET], valid_pred)
    print(auc_score)

    imp_feat = clf.feature_importances_
    ds_imp_feat = pd.Series(imp_feat, name='importance', index=features).sort_values(ascending=False)
    ds_imp_feat.plot(kind='barh')
    plt.savefig(figure_dir_path + 'imp_feat.png')

    return ds_imp_feat


def lss_model(train, valid, test):
    params = {
        'alpha': [0.022, 0.021, 0.02, 0.019, 0.023, 0.024, 0.025, 0.026, 0.027, 0.029, 0.031],
        'tol': [0.0013, 0.0014, 0.001, 0.0015, 0.0011, 0.0012, 0.0016, 0.0017]
    }
    clf = Lasso(alpha=0.031, tol=0.01, random_state=0, selection='random')
    cv = GridSearchCV(estimator=clf, param_grid=params, scoring='roc_auc', n_jobs=-1, cv=20, verbose=1)
    cv.fit(train[top_features], train[TARGET])
    print(cv.best_params_)

    valid_pred = cv.predict(valid[top_features])
    auc_score = roc_auc_score(valid[TARGET], valid_pred)
    print(auc_score)

    test_pred = cv.predict(test[top_features])

    return test_pred, auc_score


def tree_model(train, valid, test):
    params = {
        'max_depth': [1, 3, 5, 7]
    }
    clf = DecisionTreeRegressor(
        criterion='mse', max_depth=None, max_features=None,
        max_leaf_nodes=None, min_impurity_decrease=0.0,
        min_impurity_split=None, min_samples_leaf=1,
        min_samples_split=2, min_weight_fraction_leaf=0.0,
        presort=False, random_state=0, splitter='best')
    print(clf.get_params().keys())
    cv = GridSearchCV(estimator=clf, param_grid=params, scoring='roc_auc', n_jobs=-1, cv=10, verbose=1)
    cv.fit(train[top_features], train[TARGET])
    print(cv.best_params_)

    valid_pred = cv.predict(valid[top_features])
    auc_score = roc_auc_score(valid[TARGET], valid_pred)
    print(auc_score)

    test_pred = cv.predict(test[top_features])

    return test_pred, auc_score


if __name__ == '__main__':
    #
    # Load Data File
    #

    with multiprocessing.Pool() as pool:
        df_train, df_test = pool.map(load_file, ["train", "test"])

    features = [c for c in df_train.columns if c not in [ID, TARGET]]

    #
    # PreProcessing
    #

    df_train[features] = StandardScaler().fit_transform(df_train[features])
    df_test[features] = StandardScaler().fit_transform(df_test[features])

    df_train, df_valid = train_test_split(df_train, test_size=0.1, random_state=0)
    print('train shape: ', df_train.shape)
    print('valid shape: ', df_valid.shape)
    print('test shape: ', df_test.shape)

    #
    # Top Features (Logistic Regression)
    #

    lgr_clf = LogisticRegression(penalty='l1', C=0.1, random_state=0, solver='warn')
    lgr_clf.fit(df_train[features], df_train[TARGET])
    print(roc_auc_score(df_valid[TARGET], lgr_clf.predict(df_valid[features])))
    PI = PermutationImportance(estimator=lgr_clf, scoring='roc_auc').fit(df_train[features], df_train[TARGET])
    top_features = [i[1:] for i in eli5.formatters.as_dataframe.explain_weights_df(PI).feature if 'BIAS' not in i]

    #
    # Random Forest output importance features
    #


    #
    # Logistic Regression output score and predict
    #

    lgr_pred, lgr_score = lgr_model(df_train, df_valid, df_test)
    output_submit_csv(lgr_pred, lgr_score, df_test[ID], modelname='lgr')

    nn_pred, nn_score = nn_model(df_train, df_valid, df_test)
    output_submit_csv(nn_pred, nn_score, df_test[ID], modelname='nn')

    lss_pred, lss_score = lss_model(df_train, df_valid, df_test)
    output_submit_csv(lss_pred, lss_score, df_test[ID], modelname='lss')

    tree_pred, tree_score = tree_model(df_train, df_valid, df_test)
    output_submit_csv(tree_pred, tree_score, df_test[ID], modelname='tree')
