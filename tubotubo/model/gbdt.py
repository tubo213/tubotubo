from typing import List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoost, Pool
from tubotubo.model.basemodel import GBDTModel
from typing_extensions import Literal


class LGBMModel(GBDTModel):
    def __init__(self, model_params: dict, train_params: dict, cat_cols: List[str]):
        self.model_params = model_params
        self.train_params = train_params
        self.cat_cols = cat_cols

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ):
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

    def predict(self, model: lgb.Booster, X: pd.DataFrame):
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

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ):
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

    def predict(self, model: xgb.Booster, X: pd.DataFrame):
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

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ):
        train_ds = Pool(X_train, y_train, cat_features=self.cat_cols)
        valid_ds = Pool(X_valid, y_valid, cat_features=self.cat_cols)

        model = CatBoost(self.model_params)
        model.fit(train_ds, eval_set=[valid_ds], **self.train_params)
        return model

    def predict(self, model: CatBoost, X: pd.DataFrame):
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
