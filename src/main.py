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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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


# Logistic Regression (plot score and coef)
def lgr_model(train, valid, test):
    params = dict(C=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                  penalty=['l2'],
                  solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                  )
    clf = LogisticRegression(class_weight='balanced', random_state=0)
    cv = GridSearchCV(estimator=clf, param_grid=params, scoring='roc_auc', n_jobs=-1, cv=5, verbose=1)
    cv.fit(train[features], train[TARGET])
    print(cv.best_params_)

    valid_pred = cv.predict(valid[features])
    auc_score = roc_auc_score(valid[TARGET], valid_pred)
    print(auc_score)
    print(classification_report(valid[TARGET], valid_pred))

    test_pred = cv.predict(test[features])

    return test_pred, auc_score


# RandomForest Classification (plot score and return importance features)
def rfc_model(train, valid):
    clf = RandomForestClassifier(n_estimators=500, class_weight='balanced', max_depth=5, random_state=0, )
    clf.fit(train[features], train[TARGET])

    valid_pred = clf.predict(valid[features])
    auc_score = roc_auc_score(valid[TARGET], valid_pred)
    print(auc_score)

    imp_feat = clf.feature_importances_
    ds_imp_feat = pd.Series(imp_feat, name='importance', index=features).sort_values(ascending=False)
    ds_imp_feat.plot(kind='barh')
    plt.savefig(figure_dir_path + 'imp_feat.png')

    return ds_imp_feat


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
    # Random Forest output importance features
    #

    features_importance = rfc_model(df_train, df_valid)
    features = features_importance.head(10).index

    #
    # Logistic Regression output score and predict
    #

    lgr_pred, lgr_score = lgr_model(df_train, df_valid, df_test)
    output_submit_csv(lgr_pred, lgr_score, df_test[ID], modelname='lgr')
