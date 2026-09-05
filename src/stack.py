"""Combinaison de plusieurs familles de modeles (moyenne ponderee ou stacking).

PRINCIPE
--------
Chaque validation croisee produit deux artefacts sur les MEMES 3 773 images :
  outputs/<tag>_crossval/oof_probs.npy          (N, 5) predictions hors-pli
  outputs/<tag>_crossval/test_probs_ensemble.npy (M, 5) predictions de test

Comme aucune image n'a ete vue en entrainement par le modele qui la predit, ces
probabilites hors-pli permettent d'apprendre honnetement comment combiner les
familles. Trois strategies sont comparees :

  moyenne   : moyenne simple des probabilites (ce que vous faites aujourd'hui)
  ponderee  : moyenne avec des poids par famille, cherches sur les hors-pli
  stacking  : une regression logistique prend les 5*K probabilites en entree
              et apprend la combinaison, y compris les correlations entre
              familles (par exemple "quand la famille A dit Horreur et la
              famille B dit Blockbuster, c'est plutot Horreur")

Le stacking est plus expressif mais peut surapprendre. Le script mesure donc
chaque strategie par validation croisee a 5 plis SUR LES HORS-PLI, et vous
indique laquelle retenir. Ne choisissez jamais sur le score ajuste.

USAGE
-----
    python -m src.stack \\
        --oof outputs/cv_final_crossval outputs/cv_resnet_crossval \\
        --test-list data/test_list.csv \\
        --output submissions/stack.csv
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .data import read_test_list
from .metrics import format_report, macro_f1
from .predict import validate_submission, write_submission
from .utils import setup_logging

NUM_CLASSES = 5


# ---------------------------------------------------------------- chargement
def load_family(directory: Path, logger):
    """Charge (oof_probs, oof_labels, test_probs) d'une validation croisee."""
    oof = np.load(directory / "oof_probs.npy")
    labels = np.load(directory / "oof_labels.npy")
    test_path = directory / "test_probs_ensemble.npy"
    test = np.load(test_path) if test_path.exists() else None
    logger.info("%-32s hors-pli %s | macro-F1 seul = %.4f%s",
                directory.name, oof.shape, macro_f1(labels, oof.argmax(1)),
                "" if test is not None else "  (pas de probabilites de test)")
    return oof, labels, test


# ------------------------------------------------------------- combinaisons
def weighted_average(matrices: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    return np.tensordot(weights, np.stack(matrices), axes=(0, 0))


def search_weights(matrices: list[np.ndarray], y: np.ndarray,
                   step: float = 0.1) -> np.ndarray:
    """Recherche exhaustive sur une grille simplexe. Exact pour K <= 4 familles."""
    k = len(matrices)
    grid = np.arange(0, 1 + 1e-9, step)
    best_w, best_score = np.ones(k) / k, -1.0
    for combo in itertools.product(grid, repeat=k):
        total = sum(combo)
        if total <= 0:
            continue
        w = np.array(combo) / total
        score = macro_f1(y, weighted_average(matrices, w).argmax(1), NUM_CLASSES)
        if score > best_score:
            best_score, best_w = score, w
    return best_w


def fit_stacker(features: np.ndarray, y: np.ndarray, seed: int = 42):
    """Regression logistique sur les probabilites concatenees des familles."""
    return LogisticRegression(
        max_iter=3000, C=1.0, random_state=seed
    ).fit(features, y)


# ---------------------------------------------------- evaluation honnete
def honest_scores(matrices: list[np.ndarray], y: np.ndarray, folds: int = 5,
                  seed: int = 0) -> dict[str, float]:
    """Compare les trois strategies par validation croisee sur les hors-pli.

    Les poids et le stacker sont ajustes sur k-1 plis et evalues sur le pli
    restant : c'est la seule facon de savoir si le gain est reel.
    """
    features = np.hstack(matrices)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    scores = {"moyenne": [], "ponderee": [], "stacking": []}
    for train_idx, test_idx in skf.split(features, y):
        sub_train = [m[train_idx] for m in matrices]
        sub_test = [m[test_idx] for m in matrices]

        scores["moyenne"].append(
            macro_f1(y[test_idx], weighted_average(sub_test, np.ones(len(matrices))
                                                   ).argmax(1), NUM_CLASSES))

        w = search_weights(sub_train, y[train_idx])
        scores["ponderee"].append(
            macro_f1(y[test_idx], weighted_average(sub_test, w).argmax(1),
                     NUM_CLASSES))

        stacker = fit_stacker(features[train_idx], y[train_idx])
        scores["stacking"].append(
            macro_f1(y[test_idx], stacker.predict(features[test_idx]), NUM_CLASSES))

    return {k: float(np.mean(v)) for k, v in scores.items()}


# -------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="Combinaison de familles de modeles")
    parser.add_argument("--oof", nargs="+", required=True,
                        help="Dossiers <tag>_crossval, un par famille")
    parser.add_argument("--test-list", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--strategy", default="auto",
                        choices=["auto", "moyenne", "ponderee", "stacking"],
                        help="auto = la meilleure en validation croisee")
    args = parser.parse_args()

    logger = setup_logging()

    matrices, tests, labels = [], [], None
    for directory in args.oof:
        oof, y, test = load_family(Path(directory), logger)
        if labels is None:
            labels = y
        elif not np.array_equal(labels, y):
            raise SystemExit(f"{directory} : les etiquettes hors-pli different. "
                             "Les familles doivent partager la meme graine "
                             "et le meme decoupage.")
        matrices.append(oof)
        tests.append(test)

    if len(matrices) < 2:
        raise SystemExit("Il faut au moins deux familles pour combiner.")

    logger.info("=" * 62)
    logger.info("COMPARAISON HONNETE DES STRATEGIES (%d plis sur les hors-pli)",
                args.folds)
    logger.info("=" * 62)
    scores = honest_scores(matrices, labels, args.folds)
    for name, value in sorted(scores.items(), key=lambda kv: -kv[1]):
        logger.info("  %-10s macro-F1 = %.4f", name, value)

    best_strategy = (max(scores, key=scores.get) if args.strategy == "auto"
                     else args.strategy)
    gain = scores[best_strategy] - scores["moyenne"]
    logger.info("Strategie retenue : %s  (%+.4f par rapport a la moyenne simple)",
                best_strategy, gain)
    if best_strategy != "moyenne" and gain < 0.003:
        logger.warning("Gain faible : la moyenne simple est plus sure, "
                       "elle a un parametre de moins a surapprendre.")

    # --- ajustement final sur la totalite des hors-pli
    weights = None
    stacker = None
    if best_strategy == "ponderee":
        weights = search_weights(matrices, labels)
        logger.info("Poids par famille : %s",
                    {Path(d).name: round(float(w), 2)
                     for d, w in zip(args.oof, weights)})
        final_oof = weighted_average(matrices, weights)
    elif best_strategy == "stacking":
        stacker = fit_stacker(np.hstack(matrices), labels)
        final_oof = stacker.predict_proba(np.hstack(matrices))
    else:
        final_oof = weighted_average(matrices, np.ones(len(matrices)))

    logger.info("Detail hors-pli de la strategie retenue (score optimiste) :%s",
                format_report(labels, final_oof.argmax(1), NUM_CLASSES))

    # --- soumission
    if args.output:
        if any(t is None for t in tests):
            raise SystemExit("Une famille n'a pas de test_probs_ensemble.npy. "
                             "Relancez src.crossval avec --output pour la produire.")
        if not args.test_list:
            raise SystemExit("--output exige --test-list")

        if best_strategy == "ponderee":
            test_probs = weighted_average(tests, weights)
        elif best_strategy == "stacking":
            test_probs = stacker.predict_proba(np.hstack(tests))
        else:
            test_probs = weighted_average(tests, np.ones(len(tests)))

        names = read_test_list(args.test_list)
        preds = test_probs.argmax(1).tolist()
        if len(names) != len(preds):
            raise SystemExit(f"{len(names)} noms mais {len(preds)} predictions")

        write_submission(names, preds, args.output)
        if not validate_submission(args.output, names, logger):
            raise SystemExit("Soumission invalide, ne la rendez pas.")
        logger.info("Distribution : %s",
                    np.bincount(preds, minlength=NUM_CLASSES).tolist())
        logger.info("Soumission ecrite : %s", args.output)


if __name__ == "__main__":
    main()
