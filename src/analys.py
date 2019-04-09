import pandas as pd
import pandas_profiling as pdp
import numpy as np
import os
import setting
import my_path
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns

data_dir_path = my_path.DATA_DIR_PATH
profile_dir_path = my_path.PROFILE_DIR_PATH
dtypes = setting.D_TYPES
ID = 'id'
TARGET = 'target'


# return path exist (true or false)
def check_filepath(filepath):
    return os.path.isfile(filepath)


# profiling report to html file
def output_profile(df, filename):
    filepath = profile_dir_path + filename
    if check_filepath(filepath):
        print(f'already exist file path <{filepath}>')
    else:
        profile = pdp.ProfileReport(df)
        profile.to_file(filepath)
        print(f'create profiling report <{filepath}>')


# dataset: (train, test)
def load_file(dataset):
    usecols = dtypes.keys()
    if dataset == 'test':
        usecols = [col for col in dtypes.keys() if col != TARGET]
    df = pd.read_csv(data_dir_path + f'{dataset}.csv', encoding='utf-8', dtype=dtypes, usecols=usecols)
    return df


if __name__ == '__main__':
    with multiprocessing.Pool() as pool:
        df_train, df_test = pool.map(load_file, ["train", "test"])

    output_profile(df_train, 'train.html')
    output_profile(df_test, 'test.html')

    df_corr = df_train.corr()
    # sns.heatmap(df_corr, vmax=1, vmin=-1, center=0)
    # plt.show()
    # plt.savefig('../temp.png')
    # print(df_corr[TARGET].drop([ID, TARGET]).sort_values(ascending=False))
    ds_corr = df_corr[TARGET].drop([ID, TARGET])
    print(ds_corr)
    plt.show()
    plt.savefig('temp.png')
