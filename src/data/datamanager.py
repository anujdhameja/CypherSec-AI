import glob

import pandas as pd
import numpy as np
import os
import src.utils.functions.parse as parse

from os import listdir
from os.path import isfile, join
from src.utils.objects.input_dataset import InputDataset
from sklearn.model_selection import train_test_split


def read(path, json_file):
    """
    :param path: str
    :param json_file: str
    :return DataFrame
    """
    return pd.read_json(path + json_file)


def get_ratio(dataset, ratio):
    approx_size = int(len(dataset) * ratio)
    return dataset[:approx_size]


def load(path, pickle_file, ratio=1):
    dataset = pd.read_pickle(path + pickle_file)
    dataset.info(memory_usage='deep')
    if ratio < 1:
        dataset = get_ratio(dataset, ratio)

    return dataset


def write(data_frame: pd.DataFrame, path, file_name):
    data_frame.to_pickle(path + file_name)


def apply_filter(data_frame: pd.DataFrame, filter_func):
    return filter_func(data_frame)


def rename(data_frame: pd.DataFrame, old, new):
    return data_frame.rename(columns={old: new})

# def tokenize(data_frame: pd.DataFrame):
#     """
#     Tokenize the 'func' column of a DataFrame containing dicts with 'function' keys.
#     Returns a DataFrame with a single 'tokens' column.
#     """
#     def safe_tokenize(x):
#         # Check if x is a dict with 'function' key
#         if isinstance(x, dict) and 'function' in x and x['function'] is not None:
#             return parse.tokenizer(str(x['function']))
#         # fallback to empty list if missing
#         return []

#     # Apply safe tokenization
#     data_frame['func'] = data_frame['func'].apply(safe_tokenize)

#     # Rename column
#     data_frame = rename(data_frame, 'func', 'tokens')

#     # Return only tokens column
#     return data_frame[['tokens']]


def tokenize(data_frame: pd.DataFrame):
    """
    Convert function dicts to code strings and tokenize them.
    Expects data_frame.func to contain either a string or dict with 'function' key.
    """
    import src.utils.functions.parse as parse
    import pandas as pd

    def safe_tokenize(f):
        if isinstance(f, dict):
            code = f.get("function", "")
        else:
            code = str(f)
        return parse.tokenizer(code)

    # Apply safe_tokenize to all rows
    data_frame['tokens'] = data_frame['func'].apply(safe_tokenize)

    # Keep only rows with tokens
    data_frame = data_frame[data_frame['tokens'].map(len) > 0]

    # Return only tokens column
    return data_frame[["tokens"]]



def to_files(data_frame: pd.DataFrame, out_path):
    # path = f"{self.out_path}/{self.dataset_name}/"
    os.makedirs(out_path)

    for idx, row in data_frame.iterrows():
        file_name = f"{idx}.c"
        with open(out_path + file_name, 'w') as f:
            f.write(row.func)


def create_with_index(data, columns):
    data_frame = pd.DataFrame(data, columns=columns)
    data_frame.index = list(data_frame["Index"])

    return data_frame


def inner_join_by_index(df1, df2):
    return pd.merge(df1, df2, left_index=True, right_index=True)


def train_val_test_split(data_frame: pd.DataFrame, shuffle=True):
    print("Splitting Dataset")

    false = data_frame[data_frame.target == 0]
    true = data_frame[data_frame.target == 1]

    train_false, test_false = train_test_split(false, test_size=0.2, shuffle=shuffle)
    test_false, val_false = train_test_split(test_false, test_size=0.5, shuffle=shuffle)
    train_true, test_true = train_test_split(true, test_size=0.2, shuffle=shuffle)
    test_true, val_true = train_test_split(test_true, test_size=0.5, shuffle=shuffle)

    # train = train_false.append(train_true)
    # train = pd.concat([train_false, train_true])
    # val = val_false.append(val_true)
    # test = test_false.append(test_true)
    train = pd.concat([train_false, train_true])
    val = pd.concat([val_false, val_true])
    test = pd.concat([test_false, test_true])

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return InputDataset(train), InputDataset(test), InputDataset(val)


def get_directory_files(directory):
    return [os.path.basename(file) for file in glob.glob(f"{directory}/*.pkl")]

# deleted a function for our own purpose
# def loads(data_sets_dir, ratio=1):
#     data_sets_files = sorted([f for f in listdir(data_sets_dir) if isfile(join(data_sets_dir, f))])

#     if ratio < 1:
#         data_sets_files = get_ratio(data_sets_files, ratio)

#     dataset = load(data_sets_dir, data_sets_files[0])
#     data_sets_files.remove(data_sets_files[0])

#     for ds_file in data_sets_files:
#         dataset = dataset.append(load(data_sets_dir, ds_file))

#     return dataset
def loads(data_sets_dir):
    data_sets = get_directory_files(data_sets_dir)
    df_list = []
    for ds_file in data_sets:
        df_list.append(load(data_sets_dir, ds_file))

    # Use the modern pd.concat to combine all datasets at once
    dataset = pd.concat(df_list, ignore_index=True)

    print(dataset.info(verbose=True))
    return dataset

def clean(data_frame: pd.DataFrame):
    return data_frame.drop_duplicates(subset="func", keep=False)


def drop(data_frame: pd.DataFrame, keys):
    for key in keys:
        del data_frame[key]


def slice_frame(data_frame: pd.DataFrame, size: int):
    data_frame_size = len(data_frame)
    return data_frame.groupby(np.arange(data_frame_size) // size)
