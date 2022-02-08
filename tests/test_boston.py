import shutil
from pathlib import Path

import pandas as pd
from box import Box
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split
from tubotubo.model.gbdt import CatModel, LGBMModel, XGBModel
from tubotubo.preprocessing.feature_enginnearing.base import (
    AggBlock,
    CountEncodingBlock,
    IdentityBlock,
    LabelEncodingBlock,
    TargetEncodingBlock,
    run_blocks,
)
from tubotubo.utils import seed_everything

SEED = 3090
TASK_TYPE = "regression"
CFG = Box(
    {
        "seed": SEED,
        "new": False,
        "n_split": 5,
        "input_dir": Path("./input"),
        "output_dir": Path("./Output"),
        "feature_dir": Path("./featurestore/boston"),
        "target_col": "MEDV",
        "cat_cols": ["CHAS"],
        "numeric_cols": [
            "CRIM",
            "ZN",
            "INDUS",
            "NOX",
            "RM",
            "AGE",
            "DIS",
            "RAD",
            "TAX",
            "PTRATIO",
            "B",
            "LSTAT",
        ],
        "agg_funcs": ["min", "mean", "max", "std"],
        "cv_params": {
            "task_type": TASK_TYPE,
            "if_exists": "load",
            "train_fold": [0, 1, 2, 3, 4],
            "save_dir": Path("./Output/boston"),
        },
        "lgbm": {
            "model_params": {
                "boosting_type": "gbdt",
                "objective": "regression",  # rmse, mae, huber, fair, poisson, quantile, mape, gamma, tweedie
                "learning_rate": 0.01,
                "num_leaves": 64,
                "max_depth": 5,
                "seed": SEED,
                "bagging_seed": SEED,
                "feature_fraction_seed": SEED,
                "drop_seed": SEED,
                "verbose": -1,
            },
            "train_params": {
                "num_boost_round": 100000,
                "early_stopping_rounds": 40,
                "verbose_eval": 1000,
            },
            "cat_cols": ["CHAS@LabelEncodingBlock"],
        },
        "xgb": {
            "model_params": {
                "objective": "reg:squarederror",
                "max_depth": 5,
                "eta": 0.1,
                "min_child_weight": 1.0,
                "gamma": 0.0,
                "colsample_bytree": 0.8,
                "subsample": 0.8,
            },
            "train_params": {
                "num_boost_round": 100000,
                "early_stopping_rounds": 40,
                "verbose_eval": 100,
            },
        },
        "cat": {
            "model_params": {
                "loss_function": "RMSE",
                "num_boost_round": 10000,
                "depth": 7,
            },
            "train_params": {
                "early_stopping_rounds": 40,
                "plot": False,
                "use_best_model": True,
                "verbose": 100,
            },
            "cat_cols": ["CHAS@LabelEncodingBlock"],
            "task_type": TASK_TYPE,
        },
    }
)


def test_boston():
    seed_everything(CFG.seed)

    # laod
    boston = load_boston()
    whole_df = pd.DataFrame(boston.data, columns=boston.feature_names)
    whole_df["MEDV"] = boston.target

    train_df, test_df = train_test_split(
        whole_df, test_size=0.2, random_state=CFG.seed, shuffle=True
    )
    train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(drop=True)
    y_train = train_df[CFG.target_col]

    # validation
    kf = StratifiedKFold(CFG.n_split, shuffle=True, random_state=CFG.seed)
    cv_list = list(kf.split(train_df, y=train_df[CFG.cat_cols[0]]))

    # feature engineering
    blocks = [
        IdentityBlock(cols=CFG.numeric_cols),
        # OneHotEncodingBlock(cols=cat_cols, min_count=5),
        LabelEncodingBlock(cols=CFG.cat_cols),
        CountEncodingBlock(cols=CFG.cat_cols),
        *[
            AggBlock(key=key, values=CFG.numeric_cols, agg_funcs=CFG.agg_funcs)
            for key in CFG.cat_cols
        ],
        *[
            TargetEncodingBlock(
                col=col, target_col=CFG.target_col, agg_func=agg_func, cv_list=cv_list
            )
            for col in CFG.cat_cols
            for agg_func in CFG.agg_funcs
        ],
    ]

    train_feat_df = run_blocks(
        train_df,
        blocks=blocks,
        test=False,
        new=CFG.new,
        dataset_type="train",
        save_dir=CFG.feature_dir,
    )
    test_feat_df = run_blocks(
        test_df,
        blocks=blocks,
        test=True,
        new=CFG.new,
        dataset_type="test",
        save_dir=CFG.feature_dir,
    )

    # fit & inference
    lgbm = LGBMModel(**CFG.lgbm)
    xgb = XGBModel(**CFG.xgb)
    cat = CatModel(**CFG.cat)

    lgbm_results = lgbm.cv(
        y_train,
        train_feat_df,
        test_feat_df,
        cv_list,
        model_name="lgbm",
        **CFG.cv_params,
    )

    xgb_results = xgb.cv(
        y_train,
        train_feat_df,
        test_feat_df,
        cv_list,
        model_name="xgb",
        **CFG.cv_params,
    )
    cat_results = cat.cv(
        y_train,
        train_feat_df,
        test_feat_df,
        cv_list,
        model_name="cat",
        **CFG.cv_params,
    )

    lgbm.visualize_importance(lgbm_results["model_save_paths"])
    xgb.visualize_importance(xgb_results["model_save_paths"])
    cat.visualize_importance(cat_results["model_save_paths"])

    # evaluate
    for results, model in zip(
        [lgbm_results, xgb_results, cat_results], [lgbm, xgb, cat]
    ):
        model.visualize_importance(results["model_save_paths"])
        metric = mean_squared_error(y_train, results["oof"], squared=False)
        print(f"model={model.__class__.__name__} metric: {metric}")

    shutil.rmtree(CFG.output_dir)
    shutil.rmtree(CFG.feature_dir)