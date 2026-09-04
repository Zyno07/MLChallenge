"""Calibration des poids de decision par classe pour maximiser le macro-F1.

POURQUOI CE SCRIPT
------------------
Prendre l'argmax des probabilites revient a minimiser le taux d'erreur, ce qui
n'est PAS ce que mesure le macro-F1. Sous macro-F1, chaque classe compte pour
1/5 quel que soit son effectif : il est donc rentable d'accepter quelques faux
positifs supplementaires sur une classe rare si cela ameliore son rappel.

On cherche donc cinq coefficients multiplicatifs w_c tels que la prediction
devienne  argmax_c ( w_c * p_c )  au lieu de  argmax_c ( p_c ).
w_c > 1 rend la classe c plus facile a predire, w_c < 1 plus difficile.

ATTENTION AU SURAPPRENTISSAGE
-----------------------------
La classe Blockbuster ne compte que ~34 exemples en validation. Optimiser cinq
parametres libres dessus peut produire un gain illusoire. Le script mesure donc
d'abord le gain HONNETE par validation croisee (poids ajustes sur K-1 plis,
evalues sur le pli restant). Ne conservez la calibration que si ce gain
croise est positif et net.

USAGE
-----
    python -m src.calibrate --checkpoint outputs/colab_04_regul/best.pt --tta
    # puis, si le gain croise est convaincant :
    python -m src.predict --checkpoint ... --class-weights 0.95,1.60,1.00,1.05,0.90 ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PosterDataset, build_eval_transform, get_split
from .metrics import format_report, macro_f1
from .predict import load_checkpoint, predict_probabilities
from .utils import CLASS_NAMES, get_device, setup_logging

# Grille de recherche pour chaque poids : de 0.5 (classe penalisee) a 3.0.
GRID = np.concatenate([
    np.arange(0.50, 1.00, 0.05),
    np.arange(1.00, 2.05, 0.05),
    np.arange(2.20, 3.20, 0.20),
])


def fit_weights(probs: np.ndarray, y: np.ndarray, num_classes: int = 5,
                rounds: int = 4) -> np.ndarray:
    """Montee par coordonnees : on optimise un poids a la fois, plusieurs passes.

    Simple, deterministe, et suffisant pour cinq parametres.
    """
    weights = np.ones(num_classes)
    best = macro_f1(y, (probs * weights).argmax(1), num_classes)
    for _ in range(rounds):
        improved = False
        for c in range(num_classes):
            original = weights[c]
            for candidate in GRID:
                weights[c] = candidate
                score = macro_f1(y, (probs * weights).argmax(1), num_classes)
                if score > best + 1e-6:
                    best, original, improved = score, candidate, True
            weights[c] = original
        if not improved:
            break
    return weights


def cross_validated_gain(probs: np.ndarray, y: np.ndarray, folds: int = 5,
                         num_classes: int = 5, seed: int = 0) -> tuple[float, float]:
    """Gain honnete : poids ajustes hors du pli evalue.

    Renvoie (macro-F1 de base, macro-F1 calibre) moyennes sur les plis.
    """
    rng = np.random.default_rng(seed)
    # Plis stratifies : indispensable, sinon un pli peut ne contenir aucun
    # exemple de la classe rare.
    assignment = np.zeros(len(y), dtype=int)
    for c in range(num_classes):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        assignment[idx] = np.arange(len(idx)) % folds

    base_scores, calibrated_scores = [], []
    for k in range(folds):
        train_mask = assignment != k
        test_mask = ~train_mask
        w = fit_weights(probs[train_mask], y[train_mask], num_classes)
        base_scores.append(macro_f1(y[test_mask], probs[test_mask].argmax(1),
                                    num_classes))
        calibrated_scores.append(
            macro_f1(y[test_mask], (probs[test_mask] * w).argmax(1), num_classes))
    return float(np.mean(base_scores)), float(np.mean(calibrated_scores))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration macro-F1 par classe")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tta", action="store_true",
                        help="A activer si vous predisez aussi avec --tta")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--probs", default=None,
                        help="Fichier .npy de probabilites de validation deja calcule")
    parser.add_argument("--out", default=None,
                        help="Chemin .json ou ecrire les poids retenus")
    args = parser.parse_args()

    logger = setup_logging()
    device = get_device()
    model, cfg, score = load_checkpoint(args.checkpoint, device)
    logger.info("Checkpoint : macro-F1 val enregistre = %s",
                f"{score:.4f}" if score else "inconnu")

    # --- probabilites sur le MEME jeu de validation que l'entrainement
    _, val_df = get_split(cfg)
    y = val_df["label"].to_numpy()

    if args.probs and Path(args.probs).exists():
        probs = np.load(args.probs)
        logger.info("Probabilites rechargees depuis %s", args.probs)
    else:
        dataset = PosterDataset(val_df["filename"], val_df["label"],
                                cfg.data.images_dir, build_eval_transform(cfg))
        loader = DataLoader(dataset, batch_size=64, shuffle=False,
                            num_workers=cfg.data.num_workers)
        probs = predict_probabilities(model, loader, device, args.tta)
        logger.info("Probabilites calculees sur %d images de validation", len(y))

    num_classes = cfg.model.num_classes
    base = macro_f1(y, probs.argmax(1), num_classes)
    logger.info("Macro-F1 de base (argmax simple) : %.4f", base)

    # --- 1. gain honnete par validation croisee
    logger.info("Validation croisee a %d plis (gain honnete)...", args.folds)
    cv_base, cv_cal = cross_validated_gain(probs, y, args.folds, num_classes)
    gain = cv_cal - cv_base
    logger.info("  sans calibration : %.4f", cv_base)
    logger.info("  avec calibration : %.4f", cv_cal)
    logger.info("  GAIN CROISE      : %+.4f", gain)

    if gain < 0.005:
        logger.warning("Gain croise faible ou negatif : la calibration risque de")
        logger.warning("n'etre qu'un surapprentissage du jeu de validation.")
        logger.warning("Ne l'appliquez PAS au jeu de test.")
    else:
        logger.info("Gain croise significatif : la calibration est justifiee.")

    # --- 2. poids finaux, ajustes sur toute la validation
    weights = fit_weights(probs, y, num_classes)
    preds = (probs * weights).argmax(1)
    fitted = macro_f1(y, preds, num_classes)

    logger.info("\nPoids par classe retenus :")
    for c in range(num_classes):
        arrow = "plus predite" if weights[c] > 1.02 else (
            "moins predite" if weights[c] < 0.98 else "inchangee")
        logger.info("  %-13s w = %.2f   (%s)", CLASS_NAMES[c], weights[c], arrow)

    logger.info("Macro-F1 sur la validation ajustee : %.4f (optimiste, ne pas "
                "rapporter tel quel)", fitted)
    logger.info(format_report(y, preds, num_classes))

    weight_str = ",".join(f"{w:.2f}" for w in weights)
    logger.info("A passer a predict.py :  --class-weights %s", weight_str)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"weights": weights.tolist(), "cv_gain": gain,
                       "base_macro_f1": base}, f, indent=2)
        logger.info("Poids ecrits dans %s", args.out)


if __name__ == "__main__":
    main()