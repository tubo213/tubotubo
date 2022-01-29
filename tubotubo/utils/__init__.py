import os
import random
from time import time
from typing import Callable, Dict

import numpy as np
import pandas as pd


def decorate(s: str, decoration=None):
    if decoration is None:
        decoration = "★" * 20

    return " ".join([decoration, str(s), decoration])


class Timer:
    def __init__(
        self,
        logger=None,
        format_str="{:.3f}[s]",
        prefix=None,
        suffix=None,
        sep=" ",
        verbose=0,
    ):

        if prefix:
            format_str = str(prefix) + sep + format_str
        if suffix:
            format_str = format_str + sep + str(suffix)
        self.format_str = format_str
        self.logger = logger
        self.start = None
        self.end = None
        self.verbose = verbose

    @property
    def duration(self):
        if self.end is None:
            return 0
        return self.end - self.start

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()
        if self.verbose is None:
            return
        out_str = self.format_str.format(self.duration)
        if self.logger:
            self.logger.info(out_str)
        else:
            print(out_str)


def reduce_mem_usage(df, verbose=True):
    """DataFrameの型変換してメモリ削減するやつ"""
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    dfs = []
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    dfs.append(df[col].astype(np.int8))
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    dfs.append(df[col].astype(np.int16))
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    dfs.append(df[col].astype(np.int32))
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    dfs.append(df[col].astype(np.int64))
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    dfs.append(df[col].astype(np.float32))
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    dfs.append(df[col].astype(np.float32))
                else:
                    dfs.append(df[col].astype(np.float64))
        else:
            dfs.append(df[col])

    df_out = pd.concat(dfs, axis=1)
    if verbose:
        end_mem = df_out.memory_usage().sum() / 1024 ** 2
        num_reduction = str(100 * (start_mem - end_mem) / start_mem)
        print(
            f"Mem. usage decreased to {str(end_mem)[:3]}Mb:  {num_reduction[:2]}% reduction"  # noqa
        )
    return df_out


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def threshold_search(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    func: Callable[[np.ndarray, np.ndarray], float],
    is_higher_better: bool = True,
) -> Dict[str, float]:
    best_threshold = 0.0
    best_score = -np.inf if is_higher_better else np.inf

    for threshold in [i * 0.01 for i in range(100)]:
        score = func(y_true=y_true, y_pred=y_proba > threshold)
        if is_higher_better:
            if score > best_score:
                best_threshold = threshold
                best_score = score
        else:
            if score < best_score:
                best_threshold = threshold
                best_score = score

    search_result = {"threshold": best_threshold, "score": best_score}
    return search_result
