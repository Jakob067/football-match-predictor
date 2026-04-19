from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance


def plot_feature_importance(
    model,
    feature_names: list[str],
    X_reference: pd.DataFrame | None = None,
    y_reference: pd.Series | None = None,
    output_path: str | None = None,
) -> pd.Series:
    # Feature-Importances als Balkendiagramm visualisieren.
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif X_reference is not None and y_reference is not None:
        importance_result = permutation_importance(
            model,
            X_reference,
            y_reference,
            n_repeats=10,
            random_state=42,
        )
        values = importance_result.importances_mean
    else:
        raise ValueError(
            "Feature-Importances können für dieses Modell nur mit X_reference und y_reference berechnet werden."
        )

    importance_series = pd.Series(values, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=importance_series.values, y=importance_series.index, hue=importance_series.index, legend=False)
    plt.title("Feature-Wichtigkeit")
    plt.xlabel("Wichtigkeit")
    plt.ylabel("Feature")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return importance_series


def plot_confusion_matrix_heatmap(
    conf_matrix,
    class_labels: list[str],
    output_path: str | None = None,
) -> None:
    # Confusion Matrix als Heatmap darstellen.
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Vorhergesagte Klasse")
    plt.ylabel("Tatsächliche Klasse")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
