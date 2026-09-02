"""Baselines de machine learning classique sur descripteurs faits main.

Objectif triple :
 1. valider le pipeline de donnees avant d'investir dans le CNN ;
 2. donner un plancher de performance auquel comparer le reseau ;
 3. alimenter la "diversite des methodes" attendue dans le rapport.

Intuition des descripteurs : l'horreur est sombre, desaturee et contrastee ;
l'animation est claire, tres saturee et pauvre en textures fines ; la comedie
utilise beaucoup de blanc et de teintes chaudes. Ces signaux sont capturables
sans reseau de neurones.

Usage :
    python -m src.baselines --config configs/default.yaml
    python -m src.baselines --config configs/default.yaml --models rf svm logreg
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import read_labels, stratified_split
from .metrics import format_report, plot_confusion_matrix
from .utils import load_config, set_seed, setup_logging

FEATURE_SIZE = (128, 192)  # petite taille : les descripteurs sont globaux


def extract_features(path: Path) -> np.ndarray:
    """Descripteur ~90 dimensions decrivant couleur, luminosite et texture."""
    image = Image.open(path).convert("RGB").resize(FEATURE_SIZE)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

    features: list[np.ndarray] = []

    # 1. Histogrammes HSV (16 bins par canal) : la signature couleur globale.
    for c in range(3):
        hist, _ = np.histogram(hsv[:, :, c], bins=16, range=(0, 1), density=True)
        features.append(hist)

    # 2. Statistiques par canal RGB et HSV.
    for arr in (rgb, hsv):
        features.append(arr.mean(axis=(0, 1)))
        features.append(arr.std(axis=(0, 1)))

    # 3. Luminosite : moyenne, ecart-type, part de pixels tres sombres / tres clairs.
    features.append(np.array([
        gray.mean(), gray.std(),
        float((gray < 0.20).mean()),   # affiches d'horreur : beaucoup de noir
        float((gray > 0.80).mean()),   # comedies : beaucoup de blanc
        float(np.percentile(gray, 10)),
        float(np.percentile(gray, 90)),
    ]))

    # 4. Texture : energie du gradient (contours). Un dessin anime a des
    #    aplats larges, une photo de blockbuster une texture beaucoup plus riche.
    gy, gx = np.gradient(gray)
    magnitude = np.hypot(gx, gy)
    features.append(np.array([
        magnitude.mean(), magnitude.std(), float((magnitude > 0.10).mean())
    ]))

    # 5. Structure spatiale : luminosite et saturation moyennes par tiers vertical
    #    (le bas d'une affiche contient le bloc-titre, le haut l'illustration).
    thirds = np.array_split(gray, 3, axis=0)
    features.append(np.array([t.mean() for t in thirds]))
    thirds_sat = np.array_split(hsv[:, :, 1], 3, axis=0)
    features.append(np.array([t.mean() for t in thirds_sat]))

    # 6. Palette : proportion des 8 teintes dominantes.
    hue_hist, _ = np.histogram(hsv[:, :, 0], bins=8, range=(0, 1))
    features.append(hue_hist / hue_hist.sum())

    return np.concatenate([np.atleast_1d(f).ravel() for f in features])


def build_feature_matrix(filenames, images_dir: Path, logger) -> np.ndarray:
    rows = []
    for i, name in enumerate(filenames, 1):
        rows.append(extract_features(images_dir / name))
        if i % 500 == 0:
            logger.info("  %d / %d images traitees", i, len(filenames))
    return np.vstack(rows)


def get_models(seed: int) -> dict:
    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               C=1.0, random_state=seed),
        ),
        "rf": RandomForestClassifier(
            n_estimators=600, max_depth=None, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
        ),
        "svm": make_pipeline(
            StandardScaler(),
            SVC(C=5.0, gamma="scale", class_weight="balanced",
                probability=True, random_state=seed),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Baselines ML classiques")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", nargs="*", default=["logreg", "rf", "svm"])
    parser.add_argument("--cache", default="outputs/baseline_features.npz")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.output_dir) / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "baselines.log")
    set_seed(cfg.seed)

    df = read_labels(cfg.data.labels_csv)
    train_df, val_df = stratified_split(df, cfg.data.val_ratio, cfg.seed)
    images_dir = Path(cfg.data.images_dir)

    cache = Path(args.cache)
    if cache.exists():
        logger.info("Chargement des descripteurs depuis le cache %s", cache)
        data = np.load(cache)
        X_train, y_train, X_val, y_val = (data["X_train"], data["y_train"],
                                          data["X_val"], data["y_val"])
    else:
        logger.info("Extraction des descripteurs (train)...")
        X_train = build_feature_matrix(train_df["filename"], images_dir, logger)
        logger.info("Extraction des descripteurs (validation)...")
        X_val = build_feature_matrix(val_df["filename"], images_dir, logger)
        y_train = train_df["label"].to_numpy()
        y_val = val_df["label"].to_numpy()
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X_train=X_train, y_train=y_train,
                            X_val=X_val, y_val=y_val)
        logger.info("Descripteurs mis en cache dans %s", cache)

    logger.info("Dimension du descripteur : %d", X_train.shape[1])

    # Reference absolue : predire toujours la classe majoritaire.
    majority = np.bincount(y_train).argmax()
    baseline_acc = float((y_val == majority).mean())
    logger.info("Classe majoritaire (%d) : accuracy = %.4f, macro-F1 tres faible",
                majority, baseline_acc)

    for name in args.models:
        model = get_models(cfg.seed)[name]
        logger.info("Entrainement de %s ...", name)
        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        logger.info("%s termine en %.1f s", name, time.time() - start)
        logger.info("=== %s ===%s", name.upper(),
                    format_report(y_val, y_pred, cfg.model.num_classes))
        plot_confusion_matrix(y_val, y_pred, out_dir / f"confusion_{name}.png",
                              num_classes=cfg.model.num_classes)
        np.save(out_dir / f"proba_val_{name}.npy", model.predict_proba(X_val))

    logger.info("Figures et probabilites ecrites dans %s", out_dir)


if __name__ == "__main__":
    main()