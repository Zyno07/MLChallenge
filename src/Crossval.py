"""Validation croisee a k plis : entrainement, evaluation hors-pli, ensemble.

POURQUOI
--------
Un split unique 80/20 n'evalue que 755 images, dont 34 Blockbuster seulement.
A ce niveau, une difference de 0,08 en F1 sur cette classe correspond a trois
images qui basculent : impossible d'arbitrer entre deux configurations.

La validation croisee entraine k modeles sur k decoupages differents. Chaque
image du jeu est alors evaluee exactement une fois, par un modele qui ne l'a
jamais vue en entrainement. On obtient :
  - une evaluation sur les 3 773 images (169 Blockbuster au lieu de 34) ;
  - un ecart-type entre plis, donc un intervalle de confiance ;
  - k modeles diversifies qui forment naturellement un ensemble.

USAGE
-----
    # 1. entrainer les k plis (relancable : les plis deja faits sont sautes)
    python -m src.crossval --config configs/default.yaml --tag cv01 \
        --set data.image_width=192 data.image_height=288 model.width=48

    # 2. reagreger sans reentrainer
    python -m src.crossval --config configs/default.yaml --tag cv01 --skip-train

    # 3. soumission par ensemble des k modeles
    python -m src.crossval --config configs/default.yaml --tag cv01 --skip-train \
        --test-dir data/test_small --test-list data/test_list.csv \
        --output submissions/final.csv --tta
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (PosterDataset, build_eval_transform, build_test_loader,
                   get_split, read_labels)
from .metrics import (format_report, macro_f1, per_class_f1,
                      plot_confusion_matrix)
from .predict import (load_checkpoint, predict_probabilities,
                      validate_submission, write_submission)
from .utils import CLASS_NAMES, Config, get_device, load_config, setup_logging


def fold_run_name(tag: str, fold: int) -> str:
    return f"{tag}_fold{fold}"


# ------------------------------------------------------------ entrainement
def train_fold(config_path: str, tag: str, fold: int, n_folds: int,
               overrides: list[str], logger) -> Path:
    """Lance src.train dans un sous-processus pour le pli demande."""
    run_name = fold_run_name(tag, fold)
    cmd = [sys.executable, "-m", "src.train", "--config", config_path, "--set",
           f"run_name={run_name}", f"data.fold={fold}", f"data.n_folds={n_folds}",
           *overrides]
    logger.info("--- Pli %d/%d : %s", fold + 1, n_folds, run_name)
    start = time.time()
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"L'entrainement du pli {fold} a echoue.")
    logger.info("--- Pli %d termine en %.1f min", fold + 1, (time.time() - start) / 60)
    return Path("outputs") / run_name / "best.pt"


# ------------------------------------------------- predictions hors-pli
@torch.no_grad()
def collect_oof(cfg: Config, tag: str, n_folds: int, device, tta: bool, logger):
    """Assemble les predictions hors-pli des k modeles sur tout le jeu.

    Renvoie (probs, y_true, fold_scores) avec probs de forme (N, 5) dans
    l'ordre du fichier de labels d'origine.
    """
    df = read_labels(cfg.data.labels_csv)
    n_total = len(df)
    num_classes = cfg.model.num_classes

    probs = np.zeros((n_total, num_classes), dtype=np.float64)
    y_true = np.full(n_total, -1, dtype=np.int64)
    covered = np.zeros(n_total, dtype=bool)
    fold_scores = []

    for fold in range(n_folds):
        checkpoint = Path(cfg.output_dir) / fold_run_name(tag, fold) / "best.pt"
        if not checkpoint.exists():
            raise SystemExit(f"Checkpoint manquant : {checkpoint}")

        model, fold_cfg, _ = load_checkpoint(checkpoint, device)
        fold_cfg.data["fold"] = fold
        fold_cfg.data["n_folds"] = n_folds
        _, val_df = get_split(fold_cfg)

        dataset = PosterDataset(val_df["filename"], val_df["label"],
                                fold_cfg.data.images_dir,
                                build_eval_transform(fold_cfg))
        loader = DataLoader(dataset, batch_size=64, shuffle=False,
                            num_workers=fold_cfg.data.num_workers)
        fold_probs = predict_probabilities(model, loader, device, tta)

        rows = val_df["_row"].to_numpy()
        probs[rows] = fold_probs
        y_true[rows] = val_df["label"].to_numpy()
        covered[rows] = True

        score = macro_f1(val_df["label"].to_numpy(), fold_probs.argmax(1),
                         num_classes)
        fold_scores.append(score)
        logger.info("Pli %d : %d images evaluees, macro-F1 = %.4f",
                    fold, len(val_df), score)

    if not covered.all():
        raise SystemExit(f"{(~covered).sum()} images non couvertes : plis incoherents.")
    return probs, y_true, np.array(fold_scores)


# ------------------------------------------------------- ensemble sur test
@torch.no_grad()
def predict_test_ensemble(cfg: Config, tag: str, n_folds: int, device,
                          tta: bool, test_dir: str, test_list: str, logger):
    """Moyenne les probabilites des k modeles sur le jeu de test."""
    accumulated = None
    names = None
    for fold in range(n_folds):
        checkpoint = Path(cfg.output_dir) / fold_run_name(tag, fold) / "best.pt"
        model, fold_cfg, _ = load_checkpoint(checkpoint, device)
        fold_cfg.data["test_images_dir"] = test_dir
        fold_cfg.data["test_list_csv"] = test_list
        loader, names = build_test_loader(fold_cfg)
        fold_probs = predict_probabilities(model, loader, device, tta)
        accumulated = fold_probs if accumulated is None else accumulated + fold_probs
        logger.info("Pli %d applique au jeu de test", fold)
    return accumulated / n_folds, names


# -------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="Validation croisee a k plis")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tag", required=True,
                        help="Prefixe des runs, ex: cv01 -> cv01_fold0 ... cv01_fold4")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--set", nargs="*", default=[],
                        help="Surcharges transmises a src.train")
    parser.add_argument("--skip-train", action="store_true",
                        help="N'entraine pas, reutilise les checkpoints existants")
    parser.add_argument("--force", action="store_true",
                        help="Reentraine meme si le checkpoint existe deja")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--test-list", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    out_dir = Path(cfg.output_dir) / f"{args.tag}_crossval"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "crossval.log")
    device = get_device()
    num_classes = cfg.model.num_classes

    # --- 1. entrainement des plis
    if not args.skip_train:
        logger.info("Validation croisee a %d plis, tag '%s'", args.folds, args.tag)
        for fold in range(args.folds):
            checkpoint = Path(cfg.output_dir) / fold_run_name(args.tag, fold) / "best.pt"
            if checkpoint.exists() and not args.force:
                logger.info("Pli %d deja entraine, saute (--force pour refaire)", fold)
                continue
            train_fold(args.config, args.tag, fold, args.folds, args.set, logger)

    # --- 2. agregation hors-pli
    logger.info("Assemblage des predictions hors-pli...")
    probs, y_true, fold_scores = collect_oof(cfg, args.tag, args.folds, device,
                                             args.tta, logger)
    preds = probs.argmax(1)

    oof_score = macro_f1(y_true, preds, num_classes)
    f1s = per_class_f1(y_true, preds, num_classes)

    logger.info("=" * 62)
    logger.info("RESULTAT DE LA VALIDATION CROISEE (%d images evaluees)", len(y_true))
    logger.info("=" * 62)
    logger.info("Macro-F1 par pli : %s",
                " ".join(f"{s:.4f}" for s in fold_scores))
    logger.info("Moyenne des plis : %.4f  (ecart-type %.4f)",
                fold_scores.mean(), fold_scores.std())
    logger.info("Macro-F1 hors-pli global : %.4f  <- LE CHIFFRE A RAPPORTER",
                oof_score)
    logger.info("Intervalle indicatif    : %.4f a %.4f",
                fold_scores.mean() - 2 * fold_scores.std(),
                fold_scores.mean() + 2 * fold_scores.std())
    logger.info(format_report(y_true, preds, num_classes))

    for c in range(num_classes):
        support = int((y_true == c).sum())
        logger.info("  %-13s F1 = %.4f  (%d images evaluees)",
                    CLASS_NAMES[c], f1s[c], support)

    np.save(out_dir / "oof_probs.npy", probs)
    np.save(out_dir / "oof_labels.npy", y_true)
    plot_confusion_matrix(y_true, preds, out_dir / "confusion_oof.png",
                          num_classes=num_classes)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"tag": args.tag, "folds": args.folds, "tta": args.tta,
                   "fold_scores": fold_scores.tolist(),
                   "oof_macro_f1": oof_score,
                   "per_class_f1": f1s.tolist()}, f, indent=2)
    logger.info("Artefacts hors-pli ecrits dans %s", out_dir)

    # --- 3. soumission par ensemble
    if args.output:
        if not (args.test_dir and args.test_list):
            raise SystemExit("--output exige --test-dir et --test-list")
        logger.info("Generation de la soumission par ensemble des %d plis...",
                    args.folds)
        test_probs, names = predict_test_ensemble(
            cfg, args.tag, args.folds, device, args.tta,
            args.test_dir, args.test_list, logger)
        test_preds = test_probs.argmax(1).tolist()
        write_submission(names, test_preds, args.output)
        if not validate_submission(args.output, names, logger):
            raise SystemExit("Soumission invalide, ne la rendez pas.")
        np.save(out_dir / "test_probs_ensemble.npy", test_probs)
        logger.info("Distribution des predictions : %s",
                    np.bincount(test_preds, minlength=num_classes).tolist())
        logger.info("Soumission ecrite : %s", args.output)


if __name__ == "__main__":
    main()