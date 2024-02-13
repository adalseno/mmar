"""
Helper functions


"""
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
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def display_report(y_test: np.ndarray, predictions: np.ndarray) -> None:
    """Display classification report and confusion matrix

    Args:
        y_test (np.ndarray): true values
        predictions (np.ndarray): predicted values
    """
    print(classification_report(y_test, predictions))
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    return


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
    trial: optuna.Trial, X_train: pd.DataFrame, y_train: np.ndarray, metric: str
) -> float:
    """Objective function for CatBoost

    Args:
        trial (optuna.Trial): _description_
        X_train (pd.DataFrame): _description_
        y_train (np.ndarray): _description_

    Returns:
        float: _description_
    """
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
                "rf",
                CatBoostClassifier(
                    **param
                    | {
                        "used_ram_limit": "3gb",
                        "eval_metric": "Accuracy",
                        "early_stopping_rounds": 50,
                        "random_state": 1968,
                        "silent": True,
                    }
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1968)

    return cross_val_score(
        model, X_train, y_train, cv=cv, scoring=metric, n_jobs=1
    ).mean()


def objective_logistic_regression(
    trial: optuna.Trial, X_train: pd.DataFrame, y_train: np.ndarray, metric: str
) -> float:
    """Objective function for CatBoost

    Args:
        trial (optuna.Trial): _description_
        X_train (pd.DataFrame): _description_
        y_train (np.ndarray): _description_

    Returns:
        float: _description_
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
            ("rf", LogisticRegression(**param, random_state=1968)),
        ]
    )

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=1968)

    return cross_val_score(
        model, X_train, y_train, cv=cv, scoring=metric, n_jobs=1
    ).mean()


def objective_random_forest(
    trial: optuna.Trial, X_train: pd.DataFrame, y_train: np.ndarray, metric: str
) -> float:
    """Objective function for CatBoost

    Args:
        trial (optuna.Trial): _description_
        X_train (pd.DataFrame): _description_
        y_train (np.ndarray): _description_

    Returns:
        float: _description_
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
            ("rf", RandomForestClassifier(**param, random_state=1968)),
        ]
    )

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=1968)

    return cross_val_score(
        model, X_train, y_train, cv=cv, scoring=metric, n_jobs=1
    ).mean()
