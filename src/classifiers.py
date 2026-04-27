import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score, roc_auc_score


__all__ = ["logistic_regression", "gradient_boosting"]

DEFAULT_RANDOM_STATE = 0
DEFAULT_TEST_SIZE = 0.2

DEFAULT_N_REPEATS = 10
DEFAULT_SCORING = 'f1'
DEFAULT_CLASS_WEIGHT = 'balanced'


def classification_performance(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    performance = pd.DataFrame({
        "Metric": ["accuracy", "precision", "recall", "f1", "average_precision", "roc_auc"],
        "Value": [float(accuracy), float(precision), float(recall), float(f1), float(average_precision), float(roc_auc)]
    })
    return performance


def extract_significant_factors(model, X, y, feature_names) -> pd.DataFrame:
    # Extract Significant Factors
    perm = permutation_importance(
        model, X, y,
        n_repeats=DEFAULT_N_REPEATS, scoring=DEFAULT_SCORING,
        random_state=DEFAULT_RANDOM_STATE, n_jobs=-1,
    )
    importances = perm.importances_mean
    importance_std = perm.importances_std

    # Keep only features whose importance exceeds one SD of its own noise band.
    important_mask = importances > importance_std
    important_mask &= importances > 0

    factors = pd.DataFrame({
        'Factor': feature_names[important_mask],
        'Importance': importances[important_mask],
        'Std': importance_std[important_mask],
    }).sort_values(by='Importance', ascending=False)

    return factors


def logistic_regression(X: pd.DataFrame, y: pd.Series, hyperparameters: dict) -> pd.DataFrame:
    # Config
    C = hyperparameters.get("C", 0.005)

    # Setup
    X = X.fillna(X.median())
    X = X.dropna(axis=1, how='all')
    feature_names = X.columns # Preserve from the DataFrame

    # Split 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. Fit
    model = LogisticRegression(
        C=C,
        l1_ratio=1,
        solver='liblinear',
        class_weight=DEFAULT_CLASS_WEIGHT,
        random_state=DEFAULT_RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    performance = classification_performance(y_test.to_numpy(), y_pred)

    # Extract Significant Factors
    factors = extract_significant_factors(model, X_test, y_test, feature_names)
        
    return performance, factors


def gradient_boosting(X: pd.DataFrame, y: pd.Series, hyperparameters: dict) -> pd.DataFrame:
    # Config
    learning_rate = hyperparameters.get("learning_rate", 0.1)
    max_depth = hyperparameters.get("max_depth", None)
    l2_regularization = hyperparameters.get("l2_regularization", 0.0)

    # Setup
    X = X.dropna(axis=1, how='all')
    feature_names = X.columns

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE, stratify=y
    )

    # Fit
    model = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        l2_regularization=l2_regularization,
        max_iter=100,
        early_stopping=True,
        class_weight=DEFAULT_CLASS_WEIGHT,
        random_state=DEFAULT_RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    performance = classification_performance(y_test.to_numpy(), y_pred)

    # Extract Significant Factors
    factors = extract_significant_factors(model, X_test, y_test, feature_names)

    return performance, factors

