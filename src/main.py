import pandas as pd
import pandas_profiling as pdp
import numpy as np
import os
import setting
import my_path
import multiprocessing
import lightgbm as lgb
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ID = 'id'
TARGET = 'target'
dtypes = setting.D_TYPES
data_dir_path = my_path.DATA_DIR_PATH
submit_dir_path = my_path.SUBMIT_DIR_PATH


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


def lgr_model(train, valid, test):
    clf = LogisticRegression(class_weight='balanced', random_state=0, n_jobs=-1, penalty='l1')
    clf.fit(train[features], train[TARGET])

    valid_pred = clf.predict(valid[features])
    auc_score = roc_auc_score(valid[TARGET], valid_pred)
    print(auc_score)
    print(classification_report(valid[TARGET], valid_pred))

    ds_coef = pd.Series(clf.coef_[0], name='coef', index=features)
    print(ds_coef.sort_values(ascending=False))
    test_pred = clf.predict(test[features])

    return test_pred, auc_score


def rfc_model(train, valid, test):
    clf = RandomForestClassifier(n_estimators=500, class_weight='balanced', max_depth=5, random_state=0)
    clf.fit(train[features], train[TARGET])

    valid_pred = clf.predict(valid[features])
    auc_score = roc_auc_score(valid[TARGET], valid_pred)
    print(auc_score)
    print(classification_report(valid[TARGET], valid_pred))

    imp_feat = clf.feature_importances_
    ds_imp_feat = pd.Series(imp_feat, name='importance', index=features).sort_values(ascending=False)
    print(ds_imp_feat)

    test_pred = clf.predict(test[features])

    return test_pred, auc_score


if __name__ == '__main__':
    with multiprocessing.Pool() as pool:
        df_train, df_test = pool.map(load_file, ["train", "test"])

    features = [c for c in df_train.columns if c not in [ID, TARGET]]

    df_train, df_valid = train_test_split(df_train, test_size=0.1, random_state=0)
    print(df_train.shape)
    print(df_valid.shape)

    lgr_pred, lgr_score = lgr_model(df_train, df_valid, df_test)
    output_submit_csv(lgr_pred, lgr_score, df_test[ID], modelname='lgr')

    # rfc_pred, rfc_score = rfc_model(df_train, df_valid, df_test)
    # output_submit_csv(rfc_pred, rfc_score, df_test.index, modelname='rfc')
