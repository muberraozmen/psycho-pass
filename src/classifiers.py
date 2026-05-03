import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score, roc_auc_score
import asyncio
from openrouter import OpenRouter 


__all__ = ["logistic_regression", "gradient_boosting", "llama_guard"]

DEFAULT_N_REPEATS = 10
DEFAULT_SCORING = 'f1'

API_KEY = os.environ["OPENROUTER_API_KEY"]


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
        n_repeats=DEFAULT_N_REPEATS, scoring=DEFAULT_SCORING, n_jobs=-1
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


def logistic_regression(X: pd.DataFrame, y:pd.Series, test_idx:set, hyperparameters: dict) -> pd.DataFrame:
    # Config
    C = hyperparameters.get("C", 0.005)

    # Setup
    X = X.fillna(X.median())
    X = X.dropna(axis=1, how='all')
    feature_names = X.columns # Preserve from the DataFrame

    # Split 
    is_test = X.index.isin(test_idx)
    X_train, y_train = X[~is_test], y[~is_test]
    X_test, y_test = X[is_test], y[is_test]
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. Fit
    model = LogisticRegression(
        C=C,
        l1_ratio=1,
        solver='liblinear',
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    performance = classification_performance(y_test.to_numpy(), y_pred)

    # Extract Significant Factors
    factors = extract_significant_factors(model, X_test, y_test, feature_names)
        
    return performance, factors


def gradient_boosting(X: pd.DataFrame, y: pd.Series, test_idx: set, hyperparameters: dict) -> pd.DataFrame:
    # Config
    learning_rate = hyperparameters.get("learning_rate", 0.1)
    max_depth = hyperparameters.get("max_depth", None)
    l2_regularization = hyperparameters.get("l2_regularization", 0.0)

    # Setup
    X = X.dropna(axis=1, how='all')
    feature_names = X.columns

    # Split
    is_test = X.index.isin(test_idx)
    X_train, y_train = X[~is_test], y[~is_test]
    X_test, y_test = X[is_test], y[is_test]

    # Fit
    model = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        l2_regularization=l2_regularization,
        max_iter=100,
        early_stopping=True,
        class_weight='balanced',
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    performance = classification_performance(y_test.to_numpy(), y_pred)

    # Extract Significant Factors
    factors = extract_significant_factors(model, X_test, y_test, feature_names)

    return performance, factors


async def _llama_guard_task(client, semaphore, conversation_id, messages):
    async with semaphore:
        response = await client.chat.send_async(
            model="meta-llama/llama-guard-4-12b",
            messages=messages
        )
        prediction = 1 if "unsafe" in str(response.choices[0].message.content) else 0
        return conversation_id, prediction


async def llama_guard(conversations: dict, y: pd.Series, test_idx: set):
    client = OpenRouter(api_key=API_KEY)
    semaphore = asyncio.Semaphore(32)
    
    test_conversations = {k: v for k, v in conversations.items() if k in test_idx}

    async with OpenRouter(api_key=API_KEY) as client:
        tasks = [
            _llama_guard_task(client, semaphore, cid, messages)
            for cid, messages in test_conversations.items()
        ]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    predictions = {}
    failures = 0
    for outcome in outcomes:
        if isinstance(outcome, Exception):
            failures += 1
            continue
        cid, pred = outcome
        predictions[cid] = pred

    if failures:
        # swap for self.logger.warning if you thread the logger through
        print(f"llama_guard: {failures}/{len(outcomes)} requests failed after retries")

    if not predictions:
        raise RuntimeError("llama_guard produced no predictions — all requests failed.")

    y_pred = pd.Series(predictions, name="prediction")
    results = pd.concat([y_pred, y], join="inner", axis=1)

    return classification_performance(
        results["outcome"].to_numpy(),
        results["prediction"].to_numpy(),
    )