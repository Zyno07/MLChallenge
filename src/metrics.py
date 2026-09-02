"""Metriques et figures. La metrique de reference du projet est le MACRO-F1.

Pourquoi pas l'accuracy : la classe Blockbuster represente 4,5 % du jeu.
Un modele qui ne la predit jamais atteint deja ~75 % d'accuracy tout en etant
inutilisable. Le macro-F1 moyenne le F1 de chaque classe a poids egal et
sanctionne donc immediatement ce comportement.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from .utils import CLASS_NAMES


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int],
                     num_classes: int = 5) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_f1(y_true: Sequence[int], y_pred: Sequence[int],
                 num_classes: int = 5) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    f1 = np.zeros(num_classes)
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1[c] = (2 * tp / denom) if denom > 0 else 0.0
    return f1


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int],
             num_classes: int = 5) -> float:
    return float(per_class_f1(y_true, y_pred, num_classes).mean())


def full_report(y_true: Sequence[int], y_pred: Sequence[int],
                num_classes: int = 5) -> Dict[str, float]:
    f1 = per_class_f1(y_true, y_pred, num_classes)
    report = {"accuracy": accuracy(y_true, y_pred), "macro_f1": float(f1.mean())}
    for c in range(num_classes):
        report[f"f1_{c}_{CLASS_NAMES[c].split()[0].lower()}"] = float(f1[c])
    return report


def format_report(y_true: Sequence[int], y_pred: Sequence[int],
                  num_classes: int = 5) -> str:
    """Tableau texte lisible dans la console et copiable dans le rapport."""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    f1 = per_class_f1(y_true, y_pred, num_classes)
    lines = [
        "",
        f"{'Classe':<14}{'Support':>9}{'Precision':>11}{'Rappel':>9}{'F1':>8}",
        "-" * 51,
    ]
    for c in range(num_classes):
        tp = cm[c, c]
        support = cm[c, :].sum()
        prec = tp / cm[:, c].sum() if cm[:, c].sum() else 0.0
        rec = tp / support if support else 0.0
        lines.append(f"{CLASS_NAMES[c]:<14}{support:>9}{prec:>11.3f}{rec:>9.3f}{f1[c]:>8.3f}")
    lines += [
        "-" * 51,
        f"{'Accuracy':<14}{accuracy(y_true, y_pred):>37.3f}",
        f"{'Macro-F1':<14}{f1.mean():>37.3f}",
        "",
        "Matrice de confusion (lignes = verite, colonnes = prediction)",
        "        " + "".join(f"{i:>8}" for i in range(num_classes)),
    ]
    for c in range(num_classes):
        lines.append(f"{c:>8}" + "".join(f"{v:>8}" for v in cm[c]))
    return "\n".join(lines)


# --------------------------------------------------------------- figures
def plot_confusion_matrix(y_true, y_pred, out_path: str | Path,
                          normalize: bool = True, num_classes: int = 5) -> None:
    """Figure prete a coller dans le rapport Word."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, num_classes).astype(float)
    title = "Matrice de confusion"
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
        title += " (normalisee par ligne)"

    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=150)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() or 1)
    ax.set_xticks(range(num_classes), CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticks(range(num_classes), CLASS_NAMES)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Verite terrain")
    ax.set_title(title)
    for i in range(num_classes):
        for j in range(num_classes):
            value = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, value, ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.55 else "black",
                    fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_history(history: Dict[str, list], out_path: str | Path) -> None:
    """Courbes loss et macro-F1 : la figure a mettre dans la partie Resultats."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set_xlabel("Epoque"), axes[0].set_ylabel("Loss")
    axes[0].set_title("Fonction de cout")
    axes[0].legend(), axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["val_acc"], label="accuracy val")
    axes[1].plot(epochs, history["val_macro_f1"], label="macro-F1 val")
    axes[1].set_xlabel("Epoque"), axes[1].set_ylabel("Score")
    axes[1].set_title("Performance en validation")
    axes[1].legend(), axes[1].grid(alpha=0.3)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)