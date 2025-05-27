import pandas as pd
import numpy as np
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

class CreditScoringPipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, model_params=None):
        self.model_params = model_params or {
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 6,
            'eval_metric': 'AUC',
            'random_seed': 42,
            'verbose': False,
            'early_stopping_rounds': 50
        }
        self.model = None

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        self.model = CatBoostClassifier(**self.model_params)
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val)
        )
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.07).astype(int)
