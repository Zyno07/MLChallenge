"""Ensemble : moyenne (eventuellement ponderee) des probabilites de plusieurs modeles.

Trois modeles entraines avec des seeds differentes gagnent typiquement 2 a 4
points de macro-F1 par rapport au meilleur d'entre eux, pour un cout nul en
reflexion. C'est le meilleur rapport gain/effort du projet.

Usage :
    # 1. produire les probabilites de chaque modele
    python -m src.predict --checkpoint outputs/run_a/best.pt --test-dir data/test \\
        --test-list data/test_list.csv --output /tmp/a.csv --save-probs outputs/probs/a.npy
    # 2. combiner
    python -m src.ensemble --probs outputs/probs/*.npy \\
        --test-list data/test_list.csv --output submissions/final.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import read_test_list
from .predict import validate_submission, write_submission
from .utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble de modeles")
    parser.add_argument("--probs", nargs="+", required=True,
                        help="Fichiers .npy de probabilites (N, 5)")
    parser.add_argument("--weights", nargs="*", type=float, default=None,
                        help="Poids par modele (defaut : uniformes)")
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--output", default="submissions/ensemble.csv")
    args = parser.parse_args()

    logger = setup_logging()
    matrices = [np.load(p) for p in args.probs]

    shapes = {m.shape for m in matrices}
    if len(shapes) != 1:
        raise SystemExit(f"Formes incompatibles entre les fichiers : {shapes}")

    weights = np.array(args.weights or [1.0] * len(matrices), dtype=np.float64)
    if len(weights) != len(matrices):
        raise SystemExit("Il faut autant de poids que de fichiers de probabilites")
    weights = weights / weights.sum()

    combined = np.tensordot(weights, np.stack(matrices), axes=(0, 0))
    preds = combined.argmax(1).tolist()

    names = read_test_list(args.test_list)
    if len(names) != len(preds):
        raise SystemExit(f"{len(names)} noms mais {len(preds)} predictions")

    write_submission(names, preds, args.output)
    validate_submission(args.output, names, logger)

    for path, matrix in zip(args.probs, matrices):
        agreement = (matrix.argmax(1) == np.array(preds)).mean()
        logger.info("Accord avec l'ensemble : %.3f  (%s)", agreement, Path(path).name)

    logger.info("Distribution finale : %s", np.bincount(preds, minlength=5).tolist())
    logger.info("Ensemble ecrit : %s", args.output)


if __name__ == "__main__":
    main()