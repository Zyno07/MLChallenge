"""Chargement des affiches, split stratifie, augmentation, dataloaders."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from .utils import Config, NUM_CLASSES, seed_worker

# Statistiques calculees sur le jeu d'entrainement par scripts/compute_stats.py.
# Valeurs par defaut raisonnables tant que vous ne les avez pas recalculees.
DEFAULT_MEAN = (0.485, 0.446, 0.416)
DEFAULT_STD = (0.271, 0.259, 0.263)


# --------------------------------------------------------------- lecture CSV
def read_labels(csv_path: str | Path) -> pd.DataFrame:
    """Lit train_labels.csv (2 colonnes, SANS en-tete) -> DataFrame [filename, label]."""
    df = pd.read_csv(csv_path, header=None, names=["filename", "label"])
    # Filet de securite : si le fichier avait finalement un en-tete, on le retire.
    if not str(df.iloc[0]["label"]).strip().lstrip("-").isdigit():
        df = pd.read_csv(csv_path)
        df.columns = ["filename", "label"]
    df["filename"] = df["filename"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def read_test_list(csv_path: str | Path) -> List[str]:
    """Lit la liste ORDONNEE des images de test. Une colonne (nom) ou deux."""
    df = pd.read_csv(csv_path, header=None)
    names = df.iloc[:, 0].astype(str).str.strip().tolist()
    if names and not names[0].lower().endswith((".jpg", ".jpeg", ".png")):
        names = names[1:]  # en-tete accidentel
    return names


def stratified_split(
    df: pd.DataFrame, val_ratio: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split stratifie simple : chaque classe garde la meme proportion."""
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        stratify=df["label"],
        random_state=seed,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def kfold_split(
    df: pd.DataFrame, n_folds: int, fold: int, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Decoupage en k plis stratifies. Renvoie (train, validation) du pli demande.

    La colonne `_row` conserve l'indice d'origine : indispensable pour
    reassembler les predictions hors-pli (out-of-fold) dans le bon ordre.
    """
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold doit etre dans [0, {n_folds - 1}], recu {fold}")
    df = df.copy()
    df["_row"] = np.arange(len(df))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train_idx, val_idx = list(skf.split(df, df["label"]))[fold]
    return (df.iloc[train_idx].reset_index(drop=True),
            df.iloc[val_idx].reset_index(drop=True))


def get_split(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Point d'entree unique. Bascule automatiquement selon `data.fold`.

    data.fold absent ou None  -> split unique 80/20 (mode historique)
    data.fold = 0..k-1        -> validation croisee, pli demande

    Toutes les parties du code passent par ici, ce qui garantit que
    l'entrainement, l'evaluation et la prediction voient exactement le
    meme decoupage.
    """
    df = read_labels(cfg.data.labels_csv)
    fold = cfg.data.get("fold", None)
    if fold is None:
        return stratified_split(df, cfg.data.val_ratio, cfg.seed)
    return kfold_split(df, cfg.data.get("n_folds", 5), int(fold), cfg.seed)


# ------------------------------------------------------------------ dataset
class PosterDataset(Dataset):
    """Un item = (tensor image, label int). label = -1 pour le jeu de test."""

    def __init__(
        self,
        filenames: Sequence[str],
        labels: Sequence[int] | None,
        images_dir: str | Path,
        transform: Callable | None = None,
    ) -> None:
        self.filenames = list(filenames)
        self.labels = list(labels) if labels is not None else None
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def _load_image(self, name: str) -> Image.Image:
        path = self.images_dir / name
        if not path.exists():
            # Certains jeux melangent .jpg / .png : on tente les variantes.
            for ext in (".jpg", ".jpeg", ".png"):
                alt = path.with_suffix(ext)
                if alt.exists():
                    path = alt
                    break
            else:
                raise FileNotFoundError(f"Image introuvable : {path}")
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int):
        name = self.filenames[idx]
        image = self._load_image(name)
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx] if self.labels is not None else -1
        return image, label


# ----------------------------------------------------------- transformations
def build_train_transform(cfg: Config, mean=DEFAULT_MEAN, std=DEFAULT_STD):
    a = cfg.augment
    return transforms.Compose([
        transforms.Resize((cfg.data.image_height, cfg.data.image_width)),
        transforms.RandAugment(num_ops=2, magnitude=5),
        transforms.RandomHorizontalFlip(p=a.horizontal_flip),
        transforms.RandomAffine(
            degrees=a.rotation_degrees,
            translate=(a.translate, a.translate),
            scale=(a.scale_min, a.scale_max),
            fill=0,
        ),
        transforms.ColorJitter(
            brightness=a.brightness,
            contrast=a.contrast,
            saturation=a.saturation,
            hue=a.hue,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=a.random_erasing, scale=(0.02, 0.15)),
    ])


def build_eval_transform(cfg: Config, mean=DEFAULT_MEAN, std=DEFAULT_STD):
    """Aucune augmentation : validation et test doivent etre deterministes."""
    return transforms.Compose([
        transforms.Resize((cfg.data.image_height, cfg.data.image_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ------------------------------------------------------------ poids / sampler
def class_counts(labels: Sequence[int], num_classes: int = NUM_CLASSES) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int64)
    for lab in labels:
        counts[int(lab)] += 1
    return counts


def compute_class_weights(labels: Sequence[int], mode: str = "balanced") -> torch.Tensor:
    """Poids par classe pour la CrossEntropy.

    balanced      : w_c = N / (K * n_c)        -> compense integralement
    sqrt_balanced : racine du precedent        -> compense a moitie, souvent plus stable
    none          : poids uniformes
    """
    counts = class_counts(labels).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    if mode == "none":
        weights = np.ones_like(counts)
    elif mode == "balanced":
        weights = counts.sum() / (len(counts) * counts)
    elif mode == "sqrt_balanced":
        weights = np.sqrt(counts.sum() / (len(counts) * counts))
    else:
        raise ValueError(f"class_weighting inconnu : {mode}")
    weights = weights / weights.mean()  # normalise pour ne pas changer l'echelle du loss
    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_sampler(labels: Sequence[int]) -> WeightedRandomSampler:
    """Echantillonne chaque classe avec la meme probabilite a chaque epoque."""
    counts = class_counts(labels)
    per_sample = np.array([1.0 / counts[int(lab)] for lab in labels], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.tensor(per_sample, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )


# --------------------------------------------------------------- dataloaders
def build_dataloaders(cfg: Config, mean=DEFAULT_MEAN, std=DEFAULT_STD):
    """Renvoie (train_loader, val_loader, train_labels)."""
    train_df, val_df = get_split(cfg)

    train_ds = PosterDataset(
        train_df["filename"], train_df["label"], cfg.data.images_dir,
        build_train_transform(cfg, mean, std),
    )
    val_ds = PosterDataset(
        val_df["filename"], val_df["label"], cfg.data.images_dir,
        build_eval_transform(cfg, mean, std),
    )

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    use_sampler = cfg.train.sampler == "weighted"
    sampler = build_weighted_sampler(train_df["label"].tolist()) if use_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size * 2,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_df["label"].tolist()


def build_test_loader(cfg: Config, mean=DEFAULT_MEAN, std=DEFAULT_STD):
    """Loader du jeu de test, dans l'ORDRE EXACT du fichier de liste fourni."""
    names = read_test_list(cfg.data.test_list_csv)
    ds = PosterDataset(names, None, cfg.data.test_images_dir,
                       build_eval_transform(cfg, mean, std))
    loader = DataLoader(
        ds,
        batch_size=cfg.train.batch_size * 2,
        shuffle=False,           # NE JAMAIS mettre True ici
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    return loader, names
