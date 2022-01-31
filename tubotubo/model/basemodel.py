import pickle
from abc import abstractmethod
from pathlib import Path
from typing import Any, List, Literal, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
