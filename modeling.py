from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from features import FEATURE_COLUMNS


def _build_classifier(model_type: str, random_state: int):
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=350,
            random_state=random_state,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
        )
    if model_type == "neural_net":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.0005,
            max_iter=800,
            random_state=random_state,
        )
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "Für model_type='xgboost' muss das Paket 'xgboost' installiert sein."
            ) from exc

        return XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=420,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            random_state=random_state,
        )

    raise ValueError("Ungültiger model_type. Erlaubt: random_forest, neural_net, xgboost.")


def train_model(
    feature_df: pd.DataFrame,
    model_type: str = "random_forest",
    random_state: int = 42,
) -> dict[str, Any]:
    # 80/20-Split und Training des ausgewählten Klassifikators.
    if feature_df.empty:
        raise ValueError("Keine Trainingsdaten vorhanden. Bitte Saison und Datenquelle prüfen.")

    X = feature_df[FEATURE_COLUMNS]
    y = feature_df["target"]
    if y.nunique() < 2:
        raise ValueError("Zu wenige unterschiedliche Klassen für ein robustes Training.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    model = _build_classifier(model_type, random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "model_type": model_type,
    }


def train_random_forest(feature_df: pd.DataFrame, random_state: int = 42) -> dict[str, Any]:
    return train_model(feature_df=feature_df, model_type="random_forest", random_state=random_state)
