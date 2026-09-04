"""Generation du fichier de soumission.

C'EST LE SCRIPT LE PLUS CRITIQUE DU PROJET. Un modele a 70 % de macro-F1
rendu dans un mauvais ordre vaut zero. Il doit tourner et etre teste AVANT
la reception du jeu de test.

Usage :
    # verification a blanc sur le jeu de validation (a faire des le jour 1)
    python -m src.predict --checkpoint outputs/baseline_cnn/best.pt --dry-run

    # generation reelle
    python -m src.predict --checkpoint outputs/baseline_cnn/best.pt \\
        --test-dir data/test --test-list data/test_list.csv \\
        --output submissions/dupont_martin_bernard_petit.csv --tta
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np
import torch

from .data import (build_eval_transform, build_test_loader, read_labels,
                   stratified_split, PosterDataset)
from .metrics import format_report
from .models import build_model
from .utils import Config, get_device, load_config, setup_logging

VALID_LABELS = {0, 1, 2, 3, 4}


def load_checkpoint(path: str | Path, device: torch.device):
    """Recharge modele + config figee au moment de l'entrainement."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(checkpoint["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, cfg, checkpoint.get("val_macro_f1")


@torch.no_grad()
def predict_probabilities(model, loader, device, tta: bool = False) -> np.ndarray:
    """Renvoie la matrice (N, 5) des probabilites, dans l'ordre du loader."""
    all_probs: List[np.ndarray] = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        if tta:
            # Test-Time Augmentation : on moyenne avec la version miroir.
            # Gain typique de 1 a 3 points, cout nul en temps d'entrainement.
            flipped = torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
            probs = (probs + flipped) / 2
        all_probs.append(probs.cpu().numpy())
    return np.vstack(all_probs)


def write_submission(names: List[str], labels: List[int], out_path: str | Path) -> None:
    """Ecrit exactement 2 colonnes, virgule, SANS en-tete, dans l'ordre fourni."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",", lineterminator="\n")
        for name, label in zip(names, labels):
            writer.writerow([name, int(label)])


def validate_submission(path: str | Path, expected_names: List[str], logger) -> bool:
    """Relit le fichier produit et verifie tout ce qui peut mal tourner."""
    ok = True
    with open(path, "r", encoding="utf-8") as f:
        rows = [line.rstrip("\n").split(",") for line in f if line.strip()]

    if len(rows) != len(expected_names):
        logger.error("Nombre de lignes : %d attendu, %d ecrit", len(expected_names),
                     len(rows))
        ok = False

    for i, row in enumerate(rows):
        if len(row) != 2:
            logger.error("Ligne %d : %d colonnes au lieu de 2 -> %r", i + 1, len(row), row)
            ok = False
            continue
        name, label = row[0].strip(), row[1].strip()
        if i < len(expected_names) and name != expected_names[i]:
            logger.error("Ligne %d : ordre incorrect (%r attendu, %r trouve)",
                         i + 1, expected_names[i], name)
            ok = False
        if not label.isdigit() or int(label) not in VALID_LABELS:
            logger.error("Ligne %d : label invalide %r", i + 1, label)
            ok = False

    if ok:
        logger.info("Fichier VALIDE : %d lignes, ordre conforme, labels dans 0-4",
                    len(rows))
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Generation du CSV de soumission")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--test-list", default=None,
                        help="CSV donnant l'ordre exact des images de test")
    parser.add_argument("--output", default="submissions/submission.csv")
    parser.add_argument("--tta", action="store_true", help="Test-time augmentation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Teste toute la chaine sur le jeu de validation")
    parser.add_argument("--save-probs", default=None,
                        help="Chemin .npy pour sauvegarder les probabilites (ensemble)")
    parser.add_argument("--class-weights", default=None,
                        help="5 coefficients separes par des virgules, issus de "
                             "src.calibrate (ex: 0.95,1.60,1.00,1.05,0.90)")
    args = parser.parse_args()

    class_weights = None
    if args.class_weights:
        class_weights = np.array([float(v) for v in args.class_weights.split(",")])

    logger = setup_logging()
    device = get_device()
    model, cfg, score = load_checkpoint(args.checkpoint, device)
    logger.info("Checkpoint charge (macro-F1 val = %s)",
                f"{score:.4f}" if score else "inconnu")

    # ---------------------------------------------------------- mode a blanc
    if args.dry_run:
        logger.info("MODE A BLANC : prediction sur le jeu de validation")
        df = read_labels(cfg.data.labels_csv)
        _, val_df = stratified_split(df, cfg.data.val_ratio, cfg.seed)
        dataset = PosterDataset(val_df["filename"], val_df["label"],
                                cfg.data.images_dir, build_eval_transform(cfg))
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
        probs = predict_probabilities(model, loader, device, args.tta)
        if class_weights is not None:
            logger.info("Poids de decision appliques : %s", class_weights.tolist())
            probs = probs * class_weights
        preds = probs.argmax(1).tolist()
        logger.info(format_report(val_df["label"].tolist(), preds,
                                  cfg.model.num_classes))
        names = val_df["filename"].tolist()
        write_submission(names, preds, args.output)
        validate_submission(args.output, names, logger)
        logger.info("Chaine complete verifiee. Fichier test : %s", args.output)
        return

    # -------------------------------------------------------------- mode reel
    if args.test_dir:
        cfg.data["test_images_dir"] = args.test_dir
    if args.test_list:
        cfg.data["test_list_csv"] = args.test_list

    loader, names = build_test_loader(cfg)
    logger.info("%d images de test a predire depuis %s",
                len(names), cfg.data.test_images_dir)

    probs = predict_probabilities(model, loader, device, args.tta)
    if args.save_probs:
        # On sauvegarde les probabilites BRUTES, avant ponderation, pour que
        # l'ensemble puisse recombiner proprement plusieurs modeles.
        Path(args.save_probs).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_probs, probs)
        logger.info("Probabilites brutes sauvegardees dans %s", args.save_probs)
    if class_weights is not None:
        logger.info("Poids de decision appliques : %s", class_weights.tolist())
        probs = probs * class_weights
    preds = probs.argmax(1).tolist()

    write_submission(names, preds, args.output)
    if not validate_submission(args.output, names, logger):
        raise SystemExit("Le fichier de soumission est invalide, ne le rendez pas.")

    distribution = np.bincount(preds, minlength=5)
    logger.info("Distribution des predictions : %s", distribution.tolist())
    if (distribution == 0).any():
        logger.warning("Classe(s) jamais predite(s) : %s",
                       np.where(distribution == 0)[0].tolist())

    logger.info("Soumission ecrite : %s", args.output)


if __name__ == "__main__":
    main()