import os
import pandas as pd

ROOT = "./data_GP/"

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

def get_marker_data(data, marker):
    s = str(marker)
    cols = ['elapsed_time', s + '_x', s + '_y', s + '_z', s + '_c']
    sub = data[cols]
    return sub[sub[s + '_c'] > 0]

data = get_data()
run = data['YX'][0]
data_1 = get_marker_data(run, 1)
print(data_1)
