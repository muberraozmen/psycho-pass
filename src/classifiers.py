import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

__all__ = ["logistic_regression"]


def classification_performance(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    maF1 = f1_score(y_true, y_pred, average='macro')
    miF1 = f1_score(y_true, y_pred, average='micro')
    performance = pd.DataFrame({
        "Metric": ["accuracy", "precision", "recall", "f1", "maF1", "miF1"],
        "Value": [float(accuracy), float(precision), float(recall), float(f1), float(maF1), float(miF1)]
    })
    return performance


def logistic_regression(X: pd.DataFrame, y: pd.Series, random_state: int = 0, C: float = 0.005) -> pd.DataFrame:
    # 0: Fill missing values with median
    X = X.fillna(X.median())
    X = X.dropna(axis=1, how='all')
    
    # 1. Split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # 2. Scale 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    feature_names = X.columns # Preserve from the DataFrame

    # 3. Logistic Regression with L1 
    model = LogisticRegression(class_weight='balanced', l1_ratio=1, solver='liblinear', C=C, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    
    # 4. Evaluation
    y_pred = model.predict(X_test_scaled)
    performance = classification_performance(y_test.to_numpy(), y_pred)

    # 5. Extract Significant Factors
    coeffs = model.coef_[0]
    important_mask = coeffs != 0
    
    if not any(important_mask):
        print("No significant factors found. Try increasing C.")
        return performance, None
    
    factors = pd.DataFrame({
        'Factor': feature_names[important_mask],
        'Coefficient': coeffs[important_mask]
    }).sort_values(by='Coefficient', key=abs, ascending=False)
        
    return performance, factors