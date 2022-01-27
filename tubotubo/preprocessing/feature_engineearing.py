from abc import abstractmethod
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from category_encoders import CountEncoder, OrdinalEncoder
from tubotubo.utils import Timer, decorate, reduce_mem_usage


def run_blocks(input_df, blocks, test=False):
    out_df = pd.DataFrame()

    print(decorate("start run blocks..."))

    with Timer(prefix="run test={}".format(test)):
        for block in blocks:
            with Timer(prefix="\t- {}".format(str(block))):
                if not test:
                    out_i = block.fit(input_df)
                else:
                    out_i = block.transform(input_df)

            assert len(input_df) == len(out_i), block
            assert len(input_df.columns) == len(
                set(input_df.columns)
            ), "Duplicate column names"
            name = block.__class__.__name__
            out_df = pd.concat([out_df, out_i.add_suffix(f"@{name}")], axis=1)
        out_df = reduce_mem_usage(out_df)
    return out_df


class AbstractBaseBlock:
    """
    ref: https://www.guruguru.science/competitions/16/discussions/95b7f8ec-a741-444f-933a-94c33b9e66be/ # noqa
    """

    def fit(self, input_df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(input_df)

    @abstractmethod
    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class IdentityBlock(AbstractBaseBlock):
    def __init__(self, cols: List[str]):
        self.cols = cols

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        return input_df[self.cols].copy()


class OneHotEncodingsBlock(AbstractBaseBlock):
    def __init__(self, cols: List[str], min_count: int = 0):
        self.cols = cols
        self.min_count = min_count
        self.categories = {}

    def fit(self, input_df: pd.DataFrame) -> pd.DataFrame:
        for col in self.cols:
            x = input_df[col]
            vc = x.value_counts()
            categories_i = vc[vc > self.min_count].index
            self.categories[col] = categories_i

        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        df_list = []

        for col in self.cols:
            x = input_df[col]
            cat = pd.Categorical(x, categories=self.categories[col])
            df_i = pd.get_dummies(cat)
            df_i.columns = df_i.columns.tolist()
            df_i = df_i.add_prefix(f"{col}=")
            df_list.append(df_i)

        output_df = pd.concat(df_list, axis=1).astype(int)
        return output_df


class LabelEncodingBlock(AbstractBaseBlock):
    def __init__(self, cols: List[str]):
        self.cols = cols
        self.encoder = OrdinalEncoder(cols=cols)

    def fit(self, input_df: pd.DataFrame) -> pd.DataFrame:
        self.encoder.fit(input_df[self.cols])
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = input_df[self.cols].copy()
        output_df = self.encoder.transform(output_df)
        return output_df


class CountEncodingBlock(AbstractBaseBlock):
    def __init__(
        self, cols: List[str], normalize: Union[bool, dict] = False
    ):
        self.cols = cols
        self.encoder = CountEncoder(cols, normalize=normalize)

    def fit(self, input_df: pd.DataFrame) -> pd.DataFrame:
        self.encoder.fit(input_df[self.cols])
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = input_df[self.cols].copy()
        output_df = self.encoder.transform(output_df)
        return output_df


class TargetEncodingBlock(AbstractBaseBlock):
    def __init__(
        self, col: str, target_col: str, agg_func: str, cv_list: List[tuple]
    ):
        self.col = col
        self.target_col = target_col
        self.agg_func = agg_func
        self.cv_list = cv_list
        self.col_name = f"key={self.col}_agg_func={self.agg_func}"

    def fit(self, input_df: pd.DataFrame) -> pd.DataFrame:
        _input_df = input_df.reset_index(drop=True).copy()
        output_df = _input_df.copy()
        for i, (tr_idx, val_idx) in enumerate(self.cv_list):
            group = _input_df.iloc[tr_idx].groupby(self.col)[self.target_col]
            group = getattr(group, self.agg_func)().to_dict()
            output_df.loc[val_idx, self.col_name] = _input_df.loc[
                val_idx, self.col
            ].map(group)

        self.group = _input_df.groupby(self.col)[self.target_col]
        self.group = getattr(self.group, self.agg_func)().to_dict()
        return output_df[[self.col_name]].astype(np.float)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = pd.DataFrame()
        output_df[self.col_name] = input_df[self.col].map(self.group).astype(np.float)
        return output_df


class AggBlock(AbstractBaseBlock):
    def __init__(self, key: str, values: List[str], agg_funcs: List[str]):
        self.key = key
        self.values = values
        self.agg_funcs = agg_funcs

    def fit(self, input_df: pd.DataFrame) -> pd.DataFrame:
        self.meta_df = input_df.groupby(self.key)[self.values].agg(self.agg_funcs)

        # rename
        cols_level_0 = self.meta_df.columns.droplevel(0)
        cols_level_1 = self.meta_df.columns.droplevel(1)
        new_cols = [
            f"value={cols_level_1[i]}_agg_func={cols_level_0[i]}_key={self.key}"
            for i in range(len(cols_level_1))
        ]
        self.meta_df.columns = new_cols
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = self.meta_df.copy()

        # ==pd.merge(input_df, output_df, how='left', on=self.key)
        output_df = output_df.reindex(input_df[self.key].values).reset_index(drop=True)
        return output_df
