"""Verifications a lancer AVANT le premier entrainement.

    python scripts/check_data.py --config configs/default.yaml

Controle : existence des images, images corrompues, doublons, distribution
des classes, tailles et ratios, et calcule mean/std du jeu d'entrainement.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import read_labels, stratified_split  # noqa: E402
from src.utils import CLASS_NAMES, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sample-stats", type=int, default=800,
                        help="Nombre d'images pour estimer mean/std")
    args = parser.parse_args()

    cfg = load_config(args.config)
    images_dir = Path(cfg.data.images_dir)
    df = read_labels(cfg.data.labels_csv)
    problems = 0

    print(f"\n{'='*62}\nVERIFICATION DU JEU DE DONNEES\n{'='*62}")
    print(f"Fichier de labels : {cfg.data.labels_csv}  ({len(df)} lignes)")
    print(f"Dossier d'images  : {images_dir}")

    # --- doublons
    duplicates = df[df.duplicated("filename", keep=False)]
    if len(duplicates):
        print(f"\n[!] {len(duplicates)} lignes en doublon sur le nom de fichier")
        problems += 1
    else:
        print("\n[ok] Aucun doublon de nom de fichier")

    # --- labels valides
    bad_labels = df[~df["label"].isin(range(cfg.model.num_classes))]
    if len(bad_labels):
        print(f"[!] {len(bad_labels)} labels hors de 0-{cfg.model.num_classes - 1}")
        problems += 1
    else:
        print("[ok] Tous les labels sont dans l'intervalle attendu")

    # --- distribution
    print("\nDistribution des classes")
    counts = Counter(df["label"])
    total = len(df)
    for label in sorted(counts):
        n = counts[label]
        bar = "#" * int(50 * n / max(counts.values()))
        print(f"  {label} {CLASS_NAMES[label]:<12} {n:>5}  {100*n/total:5.1f} %  {bar}")
    imbalance = max(counts.values()) / min(counts.values())
    print(f"  Ratio de desequilibre max/min : {imbalance:.1f}x")
    if imbalance > 3:
        print("  -> Utilisez class_weighting=balanced et le macro-F1 comme metrique.")

    # --- fichiers presents et lisibles
    print("\nControle des images (peut prendre une minute)...")
    missing, corrupt, sizes = [], [], []
    for name in df["filename"]:
        path = images_dir / name
        if not path.exists():
            missing.append(name)
            continue
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                sizes.append(im.size)
        except Exception:
            corrupt.append(name)

    if missing:
        print(f"[!] {len(missing)} images manquantes, ex : {missing[:5]}")
        problems += 1
    else:
        print("[ok] Toutes les images du CSV sont presentes")
    if corrupt:
        print(f"[!] {len(corrupt)} images illisibles, ex : {corrupt[:5]}")
        problems += 1
    else:
        print("[ok] Toutes les images sont lisibles")

    if sizes:
        widths = np.array([s[0] for s in sizes])
        heights = np.array([s[1] for s in sizes])
        ratios = heights / widths
        print(f"\nTailles : largeur mediane {int(np.median(widths))} px, "
              f"hauteur mediane {int(np.median(heights))} px")
        print(f"Ratio hauteur/largeur : median {np.median(ratios):.2f} "
              f"(min {ratios.min():.2f}, max {ratios.max():.2f})")
        target = cfg.data.image_height / cfg.data.image_width
        print(f"Ratio cible de la config : {target:.2f} "
              f"({cfg.data.image_width}x{cfg.data.image_height})")
        if abs(target - float(np.median(ratios))) > 0.25:
            print("  [!] Ecart notable : vos images seront deformees au resize.")

    # --- split
    train_df, val_df = stratified_split(df, cfg.data.val_ratio, cfg.seed)
    print(f"\nSplit stratifie (seed={cfg.seed}) : "
          f"{len(train_df)} train / {len(val_df)} validation")
    for label in sorted(counts):
        tr = (train_df['label'] == label).mean() * 100
        va = (val_df['label'] == label).mean() * 100
        print(f"  classe {label} : {tr:5.1f} % train | {va:5.1f} % val")

    # --- mean / std
    subset = train_df.sample(min(args.sample_stats, len(train_df)),
                             random_state=cfg.seed)
    pixels = []
    for name in subset["filename"]:
        path = images_dir / name
        if not path.exists():
            continue
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB").resize((64, 96)), np.float32) / 255.0
        pixels.append(arr.reshape(-1, 3))
    if pixels:
        stacked = np.concatenate(pixels)
        mean, std = stacked.mean(0), stacked.std(0)
        print(f"\nStatistiques de normalisation (sur {len(subset)} images du TRAIN) :")
        print(f"  DEFAULT_MEAN = ({mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f})")
        print(f"  DEFAULT_STD  = ({std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f})")
        print("  -> Reportez ces valeurs dans src/data.py")

    print(f"\n{'='*62}")
    print("AUCUN PROBLEME BLOQUANT" if problems == 0
          else f"{problems} PROBLEME(S) A CORRIGER AVANT D'ENTRAINER")
    print(f"{'='*62}\n")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()