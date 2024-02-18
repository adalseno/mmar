"""
Helper functions


"""
import os

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def display_report(
    y_test: np.ndarray, predictions: np.ndarray
) -> tuple[float, float, float, float]:
    """Display classification report and confusion matrix

    Args:
        y_test (np.ndarray): true values
        predictions (np.ndarray): predicted values

    Returns:
        tuple[float,float,float,float]: true negative, false positive, false negative, true positive
    """
    print(classification_report(y_test, predictions))
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    return cm.ravel()  # tn, fp, fn, tp


def plot_feature_imp(coefficients: np.ndarray[float], columns: list[str]) -> None:
    """Plot feature importance

    Args:
        coefficients (np.ndarray[float]): coefficients
        columns (list[str]): feature names
    """
    feature_importance = pd.DataFrame(
        {"Feature": columns, "Importance": np.abs(coefficients)}
    )
    feature_importance = feature_importance.sort_values("Importance", ascending=True)
    feature_importance.plot(x="Feature", y="Importance", kind="barh", figsize=(10, 6))
    plt.show()
    return


def backtest_strategy(
    predictions: np.ndarray[int],
    X_test: pd.DataFrame,
    spy: pd.DataFrame,
    ml_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create a basic strategy dataframe

    Args:
        predictions (np.ndarray): predictions array
        X_test (pd.DataFrame): test df
        spy (pd.DataFrame): quotation df
        ml_df (pd.DataFrame): ml df

    Returns:
        pd.DataFrame: strategy df
    """
    strategy = []
    date_idx = []
    prev_price = 0.0
    for i, prediction in enumerate(predictions):
        row = X_test.iloc[i, :].to_dict()
        quote_date = X_test.index[i]
        exp_date = pd.to_datetime(ml_df.loc[quote_date, "EXPIRE_DATE"])
        # Needed since a day in YF is missing
        try:
            final_price = spy.loc[exp_date, "Close"]
        except KeyError:
            final_price = prev_price
        prev_price = final_price
        call_price = row["C_LAST"]
        strike = row["STRIKE"]
        if prediction:  # buy
            if final_price > strike:
                profit = final_price - (strike + call_price)
            else:
                profit = -call_price
            res = {
                "strategy": prediction,
                "strike": strike,
                "call_price": row["C_LAST"],
                "exp_date": ml_df.loc[quote_date, "EXPIRE_DATE"],
                "final_price": final_price,
                "profit": profit,
                "bare": max(final_price - (strike + call_price), -call_price),
            }
        else:
            res = {
                "strategy": prediction,
                "strike": strike,
                "call_price": row["C_LAST"],
                "exp_date": ml_df.loc[quote_date, "EXPIRE_DATE"],
                "final_price": final_price,
                "profit": 0.0,
                "bare": max(final_price - (strike + call_price), -call_price),
            }
        strategy.append(res)
        date_idx.append(quote_date)

    strategy_df = pd.DataFrame(strategy, pd.Index(name="Date", data=date_idx))
    strategy_df = strategy_df.sort_index()
    strategy_df["cum_profit"] = np.cumsum(strategy_df.profit.values)
    strategy_df["cum_bare"] = np.cumsum(strategy_df.bare.values)

    return strategy_df


def plot_strategy(
    strategy_df: pd.DataFrame, model_name: str, strategy_desc: str
) -> None:
    """Plot the strategy vs bare strategy

    Args:
        strategy_df (pd.DataFrame): strategy df
        model_name (str): name of the model used
        strategy_desc (str): additional stragey description
    """
    strategy_df[["cum_profit", "cum_bare"]].plot(
        figsize=(12, 6),
        ylabel="Cumulated profit",
        title=f"Example strategy based on {model_name} predictions {strategy_desc}",
    )
    ax = plt.gca()
    text = f"Strategy cumulated profit: {strategy_df['cum_profit'].iloc[-1]:.2f} in {strategy_df['strategy'].sum()} tradings"
    plt.text(0.3, 0.85, text, transform=ax.transAxes)
    plt.legend()
    plt.show()
    return


# Adapted from https://github.com/optuna/optuna-examples/blob/main/catboost/catboost_pruning.py
def objective_catboost(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    metric: str,
    n_splits: int = 5,
) -> float:
    """Objective function for CatBoost

    Args:
        trial (optuna.Trial): optuna trial
        X_train (pd.DataFrame): train set
        y_train (np.ndarray): target
        metric (str): the metric used
        n_splits (int, optional): number of splits in the CV. Defaults to 5.

    Returns:
        float: score (with penalty)
    """
    # Translate metric
    metrics = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "f1": "F1",
        "recall": "Recall",
    }
    cat_metric = metrics.get(metric, "Accuracy")
    param = {
        "objective": trial.suggest_categorical(
            "objective", ["Logloss", "CrossEntropy"]
        ),
        "colsample_bylevel": trial.suggest_float(
            "colsample_bylevel", 0.01, 0.1, log=True
        ),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical(
            "boosting_type", ["Ordered", "Plain"]
        ),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 2, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            (
                "clf",
                CatBoostClassifier(
                    **param
                    | {
                        "used_ram_limit": "3gb",
                        "eval_metric": cat_metric,
                        "early_stopping_rounds": 50,
                        "random_state": 1968,
                        "silent": True,
                    }
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1968)
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric, n_jobs=1)
    # Penalize score dispersion
    return score.mean() * (1 - score.var())


def objective_logistic_regression(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    metric: str,
    n_splits: int = 10,
) -> float:
    """Objective function for Logistic Regression

    Args:
        trial (optuna.Trial): optuna trial
        X_train (pd.DataFrame): train set
        y_train (np.ndarray): target
        metric (str): the metric used
        n_splits (int, optional): number of splits in the CV. Defaults to 10.

    Returns:
        float: metric
    """
    param = {
        "solver": trial.suggest_categorical(
            "solver",
            ["lbfgs", "newton-cg", "newton-cholesky", "sag", "saga", "liblinear"],
        ),
        "C": trial.suggest_float("C", 0.001, 100, log=True),
        "tol": trial.suggest_float("tol", 1e-6, 1e-3),
        "max_iter": trial.suggest_categorical("max_iter", [2_000]),
    }

    if param["solver"] == "saga":
        param["penalty"] = trial.suggest_categorical("penalty", ["l1", "l2"])

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**param, random_state=1968)),
        ]
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1968)
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric, n_jobs=1)
    # Penalize score dispersion
    return score.mean() * (1 - score.var())


def objective_random_forest(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    metric: str,
    n_splits: int = 10,
) -> float:
    """Objective function for Random Forest

    Args:
        trial (optuna.Trial): optuna trial
        X_train (pd.DataFrame): train set
        y_train (np.ndarray): target
        metric (str): the metric used
        n_splits (int, optional): number of splits in the CV. Defaults to 10.

    Returns:
        float: metric
    """
    param = {
        "max_features": trial.suggest_categorical(
            "max_features",
            [None, "sqrt", "log2"],
        ),
        "max_depth": trial.suggest_int("max_depth", 2, 18),
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        "bootstrap": trial.suggest_categorical(
            "bootstrap",
            [True, False],
        ),
        "criterion": trial.suggest_categorical(
            "criterion",
            ["gini", "entropy", "log_loss"],
        ),
    }

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("clf", RandomForestClassifier(**param, random_state=1968)),
        ]
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1968)
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric, n_jobs=1)
    # Penalize score dispersion
    return score.mean() * (1 - score.var())


def objective_svc(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    metric: str,
    n_splits: int = 10,
) -> float:
    """Objective function for SVC

    Args:
        trial (optuna.Trial): optuna trial
        X_train (pd.DataFrame): train set
        y_train (np.ndarray): target
        metric (str): the metric used
        n_splits (int, optional): number of splits in the CV. Defaults to 10.

    Returns:
        float: metric
    """
    param = {
        "kernel": trial.suggest_categorical(
            "kernel",
            ["linear", "poly", "rbf", "sigmoid"],
        ),
        "degree": trial.suggest_int("degree", 2, 6),
        "C": trial.suggest_float("C", 0.01, 100, log=True),
        "gamma": trial.suggest_categorical("gamma", ["auto", "scale"]),
        "shrinking": trial.suggest_categorical(
            "shrinking",
            [True, False],
        ),
    }

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("clf", SVC(**param, random_state=1968)),
        ]
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1968)
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric, n_jobs=1)
    # Penalize score dispersion
    return score.mean() * (1 - score.var())


def select_threshold(
    proba: np.ndarray[float], target: np.ndarray[int], fpr_max: float = 0.1
) -> float:
    """Compute the best threshold given the maximum acceptable false positive rate

    Args:
        proba (np.ndarray[float]): predicted probabilities
        target (np.ndarray[int]): true values
        fpr_max (float, optional): maximum acceptable false positive rate. Defaults to 0.1.

    Returns:
        float: best threshold
    """
    # calculate roc curves
    fpr, _, thresholds = roc_curve(target, proba)
    # get the best threshold with fpr <=0.1
    best_threshold = thresholds[fpr <= fpr_max][-1]

    return best_threshold


# From https://github.com/optuna/optuna-examples/blob/main/lightgbm/lightgbm_simple.py
def objective_lightgbm(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    metric: str,
    n_splits: int = 5,
    seed: int = 1968,
    max_iter: int = 35,
    max_dep: int = 12,
) -> float:
    """Objective function for LightGBM

    Args:
        trial (optuna.Trial): optuna trial
        X_train (pd.DataFrame): train set
        y_train (np.ndarray): target
        metric (str): the metric used
        n_splits (int, optional): number of splits in the CV. Defaults to 5
        seed (int, optional): seed for Kfold. Default to 1968
        max_iter (int, optional): maximun number of iterations, to prevent overfitting. Default to 35
        max_dep (int, optional): maximum depth, to prevent overfitting. Default to 12

    Returns:
        float: mean score (with penalty)
    """
    # Limit the number of iterations and max_depth
    # when we have enough features to prevent overfitting

    param = {
        "objective": "binary",
        "metric": metric,
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": 1968,
        "early_stopping_round": 15,
        "n_estimators": trial.suggest_int(
            "n_estimators", 10, max_iter, step=5, log=False
        ),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 20.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 20.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "min_sum_hessian_in_leaf": trial.suggest_float(
            "min_sum_hessian_in_leaf", 1e-6, 10.0, log=True
        ),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0),
        "feature_fraction_bynode": trial.suggest_float(
            "feature_fraction_bynode", 0.2, 1.0
        ),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "num_grad_quant_bins": trial.suggest_int("num_grad_quant_bins", 4, 12),
        "max_depth": trial.suggest_int("max_depth", 2, max_dep),
        "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
    }
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_index, test_index in cv.split(X_train, y_train):
        X_train_set, X_test = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_set, y_test = y_train[train_index], y_train[test_index]

        # Train a LightGBM model
        model = lgb.LGBMClassifier(**param)
        model.fit(
            X_train_set,
            y_train_set,
            eval_set=(X_test, y_test),
            eval_metric=metric,
        )

        # Make predictions on the test set
        y_pred = model.predict(X_test)
        # Calculate accuracy and store it in the metrics list
        score = precision_score(y_test, y_pred)
        scores.append(score)
        # Log to see the best iteration
        """ logging.warning(
            "Best iterations: %s for trial number %s with score %s", model.best_iteration_, trial.number, np.round(score,4)
        ) """

    # Penalize score dispersion
    return np.mean(scores) * (1 - np.std(scores))


def tune_params(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    metric: str,
    timeout: int,
    max_iter: int = 35,
    max_dep: int = 12,
    n_splits: int = 5,
) -> dict:
    """Tune hyperparameters using Otuna

    Args:
        X_train (pd.DataFrame): Train set
        y_train (np.ndarray): train labels
        metric (str): metric to use
        timeout (int): timeout
        max_iter (int, optional): maximun number of iterations, to prevent overfitting. Default to 35
        max_dep (int, optional): maximum depth, to prevent overfitting. Default to 12
        n_splits (int, optional): number of splits in Kfold. Default to 5

    Returns:
        dict: tuned parameters
    """
    # Set hash seed for reproducibility
    hashseed = os.getenv("PYTHONHASHSEED")
    os.environ["PYTHONHASHSEED"] = "0"
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=1968, n_startup_trials=0),
    )
    study.optimize(
        lambda trial: objective_lightgbm(
            trial,
            X_train,
            y_train,
            metric,
            max_iter=max_iter,
            max_dep=max_dep,
            n_splits=n_splits,
        ),
        n_trials=150,
        timeout=timeout,
    )

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial
    print(f"  Number: {trial.number}")
    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    # Restore hash seed
    if hashseed is not None:
        os.environ["PYTHONHASHSEED"] = hashseed
    return trial.params
