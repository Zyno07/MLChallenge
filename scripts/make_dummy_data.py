"""Genere un faux jeu de donnees pour tester le pipeline de bout en bout.

Utile pour : verifier que tout s'execute avant de recevoir les vraies images,
et pour que les 4 membres du groupe puissent developper en parallele.

Les fausses affiches ne sont pas aleatoires : chaque classe a une signature
colorimetrique differente (horreur sombre, animation saturee, etc.), donc un
pipeline correct doit atteindre un score nettement superieur au hasard. Si
l'entrainement plafonne a 20 % sur ces donnees, le bug est dans le code, pas
dans le modele.

    python scripts/make_dummy_data.py --n-train 400 --n-test 80
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# (teinte, saturation, luminosite, bruit) approximant chaque genre
CLASS_STYLE = {
    0: dict(hue=(0.05, 0.95), sat=(0.65, 0.95), val=(0.60, 0.90), noise=0.05),  # Animation
    1: dict(hue=(0.55, 0.70), sat=(0.35, 0.60), val=(0.35, 0.65), noise=0.15),  # Blockbuster
    2: dict(hue=(0.00, 0.08), sat=(0.10, 0.40), val=(0.05, 0.30), noise=0.10),  # Horreur
    3: dict(hue=(0.10, 0.18), sat=(0.45, 0.75), val=(0.70, 0.95), noise=0.08),  # Comedie
    4: dict(hue=(0.08, 0.15), sat=(0.05, 0.25), val=(0.40, 0.70), noise=0.06),  # Art&Essai
}

# Proportions reelles observees dans train_labels.csv du challenge.
REAL_DISTRIBUTION = {0: 0.215, 1: 0.045, 2: 0.210, 3: 0.223, 4: 0.307}


def make_poster(label: int, size=(180, 270), rng=None) -> Image.Image:
    rng = rng or np.random
    style = CLASS_STYLE[label]
    w, h = size

    hue = rng.uniform(*style["hue"])
    sat = rng.uniform(*style["sat"])
    val = rng.uniform(*style["val"])

    # Degrade vertical + bruit
    gradient = np.linspace(val, val * 0.55, h)[:, None] * np.ones((1, w))
    hsv = np.zeros((h, w, 3), np.float32)
    hsv[..., 0] = (hue + rng.normal(0, 0.02, (h, w))) % 1.0
    hsv[..., 1] = np.clip(sat + rng.normal(0, 0.08, (h, w)), 0, 1)
    hsv[..., 2] = np.clip(gradient + rng.normal(0, style["noise"], (h, w)), 0, 1)

    image = Image.fromarray((hsv * 255).astype(np.uint8), mode="HSV").convert("RGB")

    # Quelques formes + un bloc-titre en bas, comme sur une vraie affiche
    draw = ImageDraw.Draw(image)
    for _ in range(rng.integers(2, 6)):
        x0, y0 = rng.integers(0, w - 30), rng.integers(0, h - 60)
        draw.ellipse([x0, y0, x0 + rng.integers(20, 70), y0 + rng.integers(20, 70)],
                     outline=(255, 255, 255), width=2)
    draw.rectangle([10, h - 55, w - 10, h - 20], fill=(20, 20, 20))
    draw.text((20, h - 45), f"GENRE {label}", fill=(240, 240, 240))
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-test", type=int, default=80)
    parser.add_argument("--root", default="data")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    root = Path(args.root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "test").mkdir(parents=True, exist_ok=True)

    labels_choices = list(REAL_DISTRIBUTION)
    probabilities = [REAL_DISTRIBUTION[c] for c in labels_choices]

    rows = []
    used_ids: set[int] = set()

    def new_id() -> int:
        while True:
            value = int(rng.integers(100000, 999999))
            if value not in used_ids:
                used_ids.add(value)
                return value

    for _ in range(args.n_train):
        label = int(rng.choice(labels_choices, p=probabilities))
        name = f"{new_id()}.jpg"
        make_poster(label, rng=rng).save(root / "images" / name, quality=88)
        rows.append((name, label))

    with open(root / "train_labels.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)

    test_rows = []
    for _ in range(args.n_test):
        label = int(rng.choice(labels_choices, p=probabilities))
        name = f"{new_id()}.jpg"
        make_poster(label, rng=rng).save(root / "test" / name, quality=88)
        test_rows.append((name, label))

    with open(root / "test_list.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerows([[n] for n, _ in test_rows])
    # Verite terrain du faux jeu de test, uniquement pour vos propres controles.
    with open(root / "test_truth.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerows(test_rows)

    print(f"Faux jeu genere dans {root}/ :")
    print(f"  {args.n_train} images d'entrainement -> {root}/images")
    print(f"  {args.n_test} images de test         -> {root}/test")
    print(f"  {root}/train_labels.csv, {root}/test_list.csv, {root}/test_truth.csv")
    print("\nRemplacez ces fichiers par les vraies donnees quand vous les recevez.")


if __name__ == "__main__":
    main()