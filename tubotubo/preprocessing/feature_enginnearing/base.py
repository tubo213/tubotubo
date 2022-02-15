import hashlib
import json
from abc import abstractmethod
from copy import copy
from pathlib import Path
from typing import List, Literal, Union

import numpy as np
import pandas as pd
from category_encoders import CountEncoder, OrdinalEncoder
from tubotubo.utils import Timer, decorate, reduce_mem_usage


def run_blocks(
    input_df: pd.DataFrame,
    blocks: list,
    test=False,
    new=True,
    dataset_type: str = "train",
    save_dir: str = "../featurestore",
):
    out_df = pd.DataFrame()

    print(decorate("start run blocks..."))

    with Timer(prefix="run test={}".format(test)):
        for block in blocks:
            with Timer(prefix="\t- {}".format(str(block))):

                # Get a unique path for each parameter.
                param = block.get_init_params()
                del param["if_exists"]
                param = json.dumps(param)
                param_hash = hashlib.sha224(param.encode()).hexdigest()
                dataset_path = (
                    Path(save_dir)
                    / block.__class__.__name__
                    / param_hash
                    / f"{dataset_type}.pkl"
                )

                if new or not (dataset_path.exists()) or (block.if_exists == "replace"):
                    if not test:
                        out_i = block.fit(input_df)
                    else:
                        out_i = block.transform(input_df)
                else:
                    with Timer(
                        prefix="\t\t- load dataet from {}".format(str(dataset_path))
                    ):
                        out_i = block.load(dataset_path)

                # save dataset
                if new or not (dataset_path.exists()):
                    block.if_exists = "replace"
                    block.save(out_i, dataset_type=dataset_type, save_dir=save_dir)
                else:
                    block.save(out_i, dataset_type=dataset_type, save_dir=save_dir)

            assert len(input_df) == len(out_i), block
            name = block.__class__.__name__
            out_df = pd.concat([out_df, out_i.add_suffix(f"@{name}")], axis=1)
        assert len(out_df.columns) == len(set(out_df.columns)), "Duplicate column names"
        out_df = reduce_mem_usage(out_df)
    return out_df


class AbstractBaseBlock:
    """
    ref: https://www.guruguru.science/competitions/16/discussions/95b7f8ec-a741-444f-933a-94c33b9e66be/ # noqa
    """

    def __init__(
        self,
        if_exists: Literal["pass", "replace", "fail"] = "pass",
    ):
        self.if_exists = if_exists

    def fit(self, input_df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(input_df)

    @abstractmethod
    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def load(self, dataset_path) -> pd.DataFrame:
        processed_df = pd.read_pickle(dataset_path)
        return processed_df

    def save(
        self,
        processed_df: pd.DataFrame,
        dataset_type: str = "train",
        save_dir: str = "../featurestore",
    ) -> None:
        param = self.get_init_params()
        del param["if_exists"]
        param = json.dumps(param)
        param_hash = hashlib.sha224(param.encode()).hexdigest()

        save_dir = Path(save_dir) / self.__class__.__name__ / param_hash
        save_dir.mkdir(parents=True, exist_ok=True)
        param_path = save_dir / "param.json"
        dataset_path = save_dir / f"{dataset_type}.pkl"

        if not (dataset_path.exists()) or (self.if_exists == "replace"):
            self._save_param_and_dataset(param, processed_df, param_path, dataset_path)
        elif dataset_path.exists() and (self.if_exists == "pass"):
            pass
        elif dataset_path.exists() and (self.if_exists == "fail"):
            raise FileExistsError

    def _save_param_and_dataset(
        self, param, processed_df, param_path: Path, dataset_path: Path
    ) -> None:
        with open(param_path, "w") as f:
            json.dump(
                json.loads(param),
                f,
                ensure_ascii=False,
                indent=4,
                separators=(",", ": "),
            )
        processed_df.to_pickle(dataset_path)

    def get_init_params(self) -> dict:
        init_param_names = self.__init__.__code__.co_varnames[
            1 : self.__init__.__code__.co_argcount
        ]
        instance_params = copy(self.__dict__)
        instance_param_names = list(instance_params.keys())
        for key in instance_param_names:
            if key not in init_param_names:
                del instance_params[key]
        return instance_params


class IdentityBlock(AbstractBaseBlock):
    def __init__(self, cols: List[str], if_exists="pass"):
        super().__init__(if_exists)
        self.cols = cols

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        return input_df[self.cols].copy()


class OneHotEncodingBlock(AbstractBaseBlock):
    def __init__(self, cols: List[str], min_count: int = 0, if_exists: str = "pass"):
        super().__init__(if_exists)
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
    def __init__(self, cols: List[str], handle_unknown='return_nan', handle_missing='return_nan', if_exists: str = "pass"):
        super().__init__(if_exists)
        self.cols = cols
        self.handle_missing = handle_missing
        self.handle_unknown = handle_unknown
        self.encoder = OrdinalEncoder(cols=cols, handle_unknown=handle_unknown, handle_missing=handle_missing)

    def fit(self, input_df: pd.DataFrame) -> pd.DataFrame:
        self.encoder.fit(input_df[self.cols])
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = input_df[self.cols].copy()
        output_df = self.encoder.transform(output_df).astype(float)
        return output_df


class CountEncodingBlock(AbstractBaseBlock):
    def __init__(
        self,
        cols: List[str],
        normalize: Union[bool, dict] = False,
        if_exists: str = "pass",
    ):
        super().__init__(if_exists)
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
        self,
        col: str,
        target_col: str,
        agg_func: str,
        cv_list: List[tuple],
        if_exists: str = "pass",
    ):
        super().__init__(if_exists)
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

    def get_init_params(self) -> dict:
        init_param_names = self.__init__.__code__.co_varnames[
            1 : self.__init__.__code__.co_argcount
        ]
        instance_params = copy(self.__dict__)
        instance_param_names = list(instance_params.keys())
        for key in instance_param_names:
            if key not in init_param_names:
                del instance_params[key]

        del instance_params["cv_list"]
        return instance_params


class AggBlock(AbstractBaseBlock):
    def __init__(
        self, key: str, values: List[str], agg_funcs: List[str], if_exists: str = "pass"
    ):
        super().__init__(if_exists)
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
