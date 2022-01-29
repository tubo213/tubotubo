import pickle
from abc import abstractmethod
from pathlib import Path
from typing import Any, List, Literal, Tuple, Union

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from catboost import CatBoost, Pool
from tubotubo.utils import Timer


class BaseModel:
    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def predict(self, model, X) -> Union[np.ndarray, np.array]:
        raise NotImplementedError

    def cv(
        self,
        y: pd.Series,
        train_feat_df: pd.DataFrame,
        test_feat_df: pd.DataFrame,
        cv_list: List[tuple],
        train_fold: List[int],
        task_type: Literal["regression", "binary", "multiclass"],
        save_dir: str = "./Output/",
        model_name: str = "lgbm",
        if_exists: Literal["replace", "load"] = "replace",
    ):
        save_dir = Path(save_dir)
        model_save_dir = save_dir / "model"
        model_save_dir.mkdir(parents=True, exist_ok=True)

        n_output_col = y.nunique()
        used_idx = []
        model_save_paths = []
        oof_pred = (
            np.zeros((len(train_feat_df), n_output_col))
            if task_type == "multiclass"
            else np.zeros(len(train_feat_df))
        )
        test_pred = (
            np.zeros((len(test_feat_df), n_output_col))
            if task_type == "multiclass"
            else np.zeros(len(test_feat_df))
        )

        with Timer(prefix="run cv"):
            for fold, (train_idx, valid_idx) in enumerate(cv_list):
                if fold in train_fold:
                    with Timer(prefix="\t- run fold {}".format(fold)):
                        fold_model_save_path = (
                            model_save_dir / f"{model_name}_fold{fold}.pkl"
                        )
                        used_idx.append(valid_idx)
                        model_save_paths.append(fold_model_save_path)

                        # split
                        X_train, y_train = (
                            train_feat_df.iloc[train_idx],
                            y.iloc[train_idx],
                        )
                        X_valid, y_valid = (
                            train_feat_df.iloc[valid_idx],
                            y.iloc[valid_idx],
                        )

                        # fit or load
                        if (if_exists == "replace") or not (
                            fold_model_save_path.exists()
                        ):
                            model = self.fit(X_train, y_train, X_valid, y_valid)
                            self.save(model, fold_model_save_path)
                        else:
                            model = self.load(fold_model_save_path)

                        # infernce
                        oof_pred[valid_idx] += self.predict(model, X_valid)
                        test_pred += self.predict(model, test_feat_df)
                else:
                    pass

            test_pred /= len(train_fold)

            results = {
                "oof": oof_pred,
                "test": test_pred,
                "used_idx": used_idx,
                "model_save_paths": model_save_paths,
            }

            return results

    def save(self, model: Any, save_path: Union[str, Path]) -> None:
        with open(save_path, "wb") as p:
            pickle.dump(model, p)

    def load(self, save_path: Union[str, Path]) -> Any:
        with open(save_path, "rb") as p:
            model = pickle.load(p)
        return model


class GBDTModel(BaseModel):
    @abstractmethod
    def _get_feature_importances(
        self, model_save_paths: List[str]
    ) -> Tuple[List[np.array], List[List[str]]]:
        raise NotImplementedError

    def visualize_importance(
        self, model_save_paths: List[str]
    ) -> Tuple[plt.Figure, plt.Axes]:
        feature_importances, columns = self._get_feature_importances(model_save_paths)
        feature_importance_df = pd.DataFrame()
        for i, feature_importance_i in enumerate(feature_importances):
            _df = pd.DataFrame()
            _df["feature_importance"] = feature_importance_i
            _df["column"] = columns[i]
            _df["fold"] = i + 1
            feature_importance_df = pd.concat(
                [feature_importance_df, _df], axis=0, ignore_index=True
            )

        order = (
            feature_importance_df.groupby("column")
            .sum()[["feature_importance"]]
            .sort_values("feature_importance", ascending=False)
            .index[:50]
        )

        fig, ax = plt.subplots(figsize=(8, max(6, len(order) * 0.25)))
        sns.boxenplot(
            data=feature_importance_df,
            x="feature_importance",
            y="column",
            order=order,
            ax=ax,
            palette="viridis",
            orient="h",
        )
        ax.tick_params(axis="x", rotation=90)
        ax.set_title("Importance")
        ax.grid()
        fig.tight_layout()
        return fig, ax


class LGBMModel(GBDTModel):
    def __init__(self, model_params: dict, train_params: dict, cat_cols: List[str]):
        self.model_params = model_params
        self.train_params = train_params
        self.cat_cols = cat_cols

    def fit(self, X_train, y_train, X_valid, y_valid):
        train_ds = lgb.Dataset(X_train, y_train, categorical_feature=self.cat_cols)
        valid_ds = lgb.Dataset(X_valid, y_valid, categorical_feature=self.cat_cols)

        model = lgb.train(
            params=self.model_params,
            train_set=train_ds,
            valid_sets=[train_ds, valid_ds],
            valid_names=["train", "valid"],
            **self.train_params,
        )
        return model

    def predict(self, model, X):
        return model.predict(X)

    def _get_feature_importances(
        self, model_save_paths: List[str]
    ) -> Tuple[List[np.array], List[List[str]]]:
        feature_importances = []
        for path in model_save_paths:
            model = self.load(path)
            feature_importance_i = model.feature_importance(importance_type="gain")
            feature_importances.append(feature_importance_i)
        columns = model.feature_name()
        columns = [columns] * len(feature_importances)
        return feature_importances, columns


class XGBModel(GBDTModel):
    def __init__(self, model_params: dict, train_params: dict):
        self.model_params = model_params
        self.train_params = train_params

    def fit(self, X_train, y_train, X_valid, y_valid):
        feature_names = X_train.columns
        train_ds = xgb.DMatrix(X_train, y_train, feature_names=feature_names)
        valid_ds = xgb.DMatrix(X_valid, y_valid, feature_names=feature_names)

        model = xgb.train(
            params=self.model_params,
            dtrain=train_ds,
            evals=[(train_ds, "train"), (valid_ds, "eval")],
            **self.train_params,
        )

        return model

    def predict(self, model, X):
        pred_ds = xgb.DMatrix(X)
        return model.predict(pred_ds, ntree_limit=model.best_ntree_limit)

    def _get_feature_importances(
        self, model_save_paths: List[str]
    ) -> Tuple[List[np.array], List[List[str]]]:
        feature_importances = []
        columns = []
        for path in model_save_paths:
            model = self.load(path)
            score = model.get_score(importance_type="gain")

            feature_importances_i = np.array(list(score.values()))
            columns_i = list(score.keys())

            feature_importances.append(feature_importances_i)
            columns.append(columns_i)

        return feature_importances, columns


class CatModel(GBDTModel):
    def __init__(
        self,
        model_params: dict,
        train_params: dict,
        cat_cols: List[str],
        task_type: Literal["regression", "binary", "multiclass"],
    ):
        self.model_params = model_params
        self.train_params = train_params
        self.cat_cols = cat_cols
        self.task_type = task_type

    def fit(self, X_train, y_train, X_valid, y_valid):
        train_ds = Pool(X_train, y_train, cat_features=self.cat_cols)
        valid_ds = Pool(X_valid, y_valid, cat_features=self.cat_cols)

        model = CatBoost(self.model_params)
        model.fit(train_ds, eval_set=[valid_ds], **self.train_params)
        return model

    def predict(self, model, X):
        if self.task_type == "regression":
            return model.predict(X)
        elif self.task_type == "binary":
            return model.predict(X, prediction_type="Probability")[:, 1]
        elif self.task_type == "multiclass":
            return model.predict(X, prediction_type="Probability")
        else:
            raise ValueError

    def _get_feature_importances(
        self, model_save_paths: List[str]
    ) -> Tuple[List[np.array], List[List[str]]]:
        feature_importances = []
        columns = []
        for path in model_save_paths:
            model = self.load(path)
            feature_importance_i = model.get_feature_importance(
                type="FeatureImportance"
            )
            columns.append(model.feature_names_)
            feature_importances.append(feature_importance_i)
        return feature_importances, columns
