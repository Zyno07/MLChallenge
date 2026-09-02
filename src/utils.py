"""Utilitaires transverses : reproductibilite, device, config, logs."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

# Noms lisibles des classes, dans l'ordre des labels du sujet.
CLASS_NAMES = ["Animation", "Blockbuster", "Horreur", "Comedie", "Art & Essai"]
NUM_CLASSES = len(CLASS_NAMES)


# ------------------------------------------------------------------ config
class Config(dict):
    """Dictionnaire accessible en pointe : cfg.train.lr au lieu de cfg['train']['lr'].

    Les sous-dictionnaires sont convertis EN PLACE a la construction. C'est
    important : si on les convertissait a la volee dans __getattr__, chaque
    acces renverrait un nouvel objet et `cfg.data["images_dir"] = x` ne
    modifierait qu'une copie temporaire, silencieusement.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            if isinstance(value, dict) and not isinstance(value, Config):
                super().__setitem__(key, Config(value))

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
        super().__setitem__(key, value)

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - message d'erreur plus clair
            raise AttributeError(f"Cle absente de la config : {item}") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Charge un YAML puis applique des surcharges type `train.lr=0.0003`."""
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Surcharge invalide (attendu cle=valeur) : {override}")
        key, value = override.split("=", 1)
        node = raw
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)  # convertit "3" -> 3, "true" -> True

    return Config(raw)


def save_config(cfg: Config, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(json.loads(json.dumps(cfg)), f, sort_keys=False,
                       allow_unicode=True)


# ------------------------------------------------------- reproductibilite
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Fixe toutes les sources d'alea. A appeler AVANT toute creation de modele."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # Plus lent mais strictement reproductible.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """Seed des workers du DataLoader (sinon l'augmentation n'est pas reproductible)."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ------------------------------------------------------------------ device
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Mac Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return device.type


# ------------------------------------------------------------------- logs
def setup_logging(log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger("challenge")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)