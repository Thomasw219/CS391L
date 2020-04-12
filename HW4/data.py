import os
import pandas as pd
import numpy as np

ROOT = "./GP_data/"

def get_data():
    dirs = os.listdir(ROOT)
    data = {}
    for d in dirs:
        root = ROOT + d + '/'
        files = []
        for (dirpath, dirnames, filenames) in os.walk(root):
            fullnames = [dirpath + '/' + name for name in filenames]
            files.extend(fullnames)

        dataframes = []
        for f in files:
            df = pd.read_csv(f)
            dataframes.append(df)

        data[d] = dataframes

    return data

def get_all_data(data, person, n_runs=5):
    arr = data[person][0:n_runs]
    df = None
    for d in arr:
#        dp = d.iloc[0::7].append(d.iloc[1::7])
        dp = d.iloc[0::3]
        if df is None:
            df = dp.copy()
        else:
            df = df.append(dp, sort=False)
    return df

def get_marker_data(data, marker, coord):
    s = str(marker)
    cols = ['elapsed_time', s + '_' + coord, s + '_c', 'frame']
    sub = data[cols]
    return sub[sub[s + '_c'] > 0]

def get_target_data(data):
    cols = ['elapsed_time', '15_x', '15_y', '15_z', '15_c']
    sub = data[cols]
    return sub[sub['15_c'] > 0]

def restrict_times(df, t_i, t_f):
    dcp = df.copy()
    temp = dcp[dcp['elapsed_time'] >= t_i]
    return temp[temp['elapsed_time'] <= t_f]

def to_matrices(df):
    mat = df.to_numpy(dtype=float)
    X = np.reshape(mat[:, 0], (mat.shape[0], 1))
    Y = np.reshape(mat[:, 1], (mat.shape[0], 1))
    return X, Y
