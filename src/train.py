"""Entrainement du CNN.

Usage :
    python -m src.train --config configs/default.yaml
    python -m src.train --config configs/default.yaml --set train.lr=0.0003 run_name=lr3e4
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from torch.optim.swa_utils import SWALR, AveragedModel, update_bn

from .data import build_dataloaders, compute_class_weights
from .metrics import (format_report, full_report, macro_f1, plot_confusion_matrix,
                      plot_history)
from .models import build_model
from .utils import (Config, count_parameters, describe_device, get_device,
                    load_config, save_config, set_seed, setup_logging)


# ----------------------------------------------------------------- mixup
def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """Melange lineaire de deux images et de leurs labels.

    Regularisation efficace quand les donnees sont peu nombreuses : le modele
    ne peut plus memoriser une image precise puisqu'il n'en voit jamais deux
    fois la meme combinaison.
    """
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[index]
    return mixed, y, y[index], lam


# ------------------------------------------------------------- scheduler
def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Cosine avec warmup lineaire, applique a chaque batch."""
    if cfg.train.scheduler == "none":
        return None
    if cfg.train.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

    total_steps = cfg.train.epochs * steps_per_epoch
    warmup_steps = cfg.train.warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ----------------------------------------------------------------- boucles
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                    device, cfg) -> float:
    model.train()
    running_loss, seen = 0.0, 0
    step_scheduler = scheduler is not None and not isinstance(
        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        images, y_a, y_b, lam = mixup_batch(images, labels, cfg.train.mixup_alpha)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,
                            enabled=cfg.train.amp and device.type == "cuda"):
            outputs = model(images)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)

        scaler.scale(loss).backward()
        if cfg.train.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if step_scheduler:
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        seen += images.size(0)

    return running_loss / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[float, List[int], List[int]]:
    model.eval()
    running_loss, seen = 0.0, 0
    all_true: List[int] = []
    all_pred: List[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        seen += images.size(0)
        all_true += labels.cpu().tolist()
        all_pred += outputs.argmax(1).cpu().tolist()

    return running_loss / max(seen, 1), all_true, all_pred


# --------------------------------------------------------------------- SWA
def finalize_swa(swa_model, train_loader, val_loader, eval_criterion, device,
                 cfg, run_dir, best_f1, logger):
    """Recalcule les statistiques de BatchNorm puis evalue le modele moyenne.

    Le Stochastic Weight Averaging moyenne les POIDS des dernieres epoques au
    lieu d'en selectionner une seule. Interet ici : le macro-F1 de validation
    oscille fortement d'une epoque a l'autre (ecart-type ~0,017 mesure en
    validation croisee), donc retenir le maximum d'une serie bruitee surestime
    la performance reelle et donne un modele arbitraire. La moyenne des poids
    est plus stable et generalise generalement mieux.

    Etape indispensable : les statistiques courantes de BatchNorm ne sont pas
    des parametres appris, elles ne peuvent donc pas etre moyennees. Il faut
    les recalculer en parcourant le jeu d'entrainement une fois.
    """
    logger.info("SWA : recalcul des statistiques de BatchNorm...")
    swa_model.to(device)
    update_bn(train_loader, swa_model, device=device)

    _, y_true, y_pred = evaluate(swa_model.module, val_loader, eval_criterion, device)
    metrics = full_report(y_true, y_pred, cfg.model.num_classes)
    swa_f1 = metrics["macro_f1"]

    torch.save({
        "model_state": swa_model.module.state_dict(),
        "config": json.loads(json.dumps(cfg)),
        "epoch": -1,
        "val_macro_f1": swa_f1,
    }, run_dir / "swa.pt")

    logger.info("SWA  macro-F1 = %.4f   |   meilleure epoque = %.4f   (ecart %+.4f)",
                swa_f1, best_f1, swa_f1 - best_f1)

    if swa_f1 > best_f1:
        torch.save({
            "model_state": swa_model.module.state_dict(),
            "config": json.loads(json.dumps(cfg)),
            "epoch": -1,
            "val_macro_f1": swa_f1,
        }, run_dir / "best.pt")
        logger.info("SWA meilleur : best.pt remplace par le modele moyenne.")
    else:
        logger.info("SWA moins bon : best.pt conserve (swa.pt reste disponible).")

    return swa_f1, y_true, y_pred, metrics


# ----------------------------------------------------- journal d'experiences
def log_experiment(cfg: Config, metrics: Dict[str, float], extra: Dict) -> None:
    """Ajoute une ligne a experiments.csv : c'est la matiere de votre rapport."""
    path = Path(cfg.experiments_log)
    row = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "run_name": cfg.run_name,
        "model": cfg.model.name,
        "width": cfg.model.width,
        "blocks": cfg.model.num_blocks,
        "dropout": cfg.model.dropout,
        "image_size": f"{cfg.data.image_width}x{cfg.data.image_height}",
        "epochs": cfg.train.epochs,
        "batch_size": cfg.train.batch_size,
        "lr": cfg.train.lr,
        "weight_decay": cfg.train.weight_decay,
        "class_weighting": cfg.train.class_weighting,
        "sampler": cfg.train.sampler,
        "mixup": cfg.train.mixup_alpha,
        "label_smoothing": cfg.train.label_smoothing,
        "seed": cfg.seed,
        **{k: round(v, 4) for k, v in metrics.items()},
        **extra,
    }
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# -------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="Entrainement CNN affiches de films")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", nargs="*", default=[],
                        help="Surcharges, ex: train.lr=0.0003 model.width=48")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    run_dir = Path(cfg.output_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(run_dir / "train.log")
    set_seed(cfg.seed)
    device = get_device()

    logger.info("Run '%s' | device : %s", cfg.run_name, describe_device(device))
    save_config(cfg, run_dir / "config.yaml")

    # --- donnees
    train_loader, val_loader, train_labels = build_dataloaders(cfg)
    logger.info("Train : %d images | Validation : %d images",
                len(train_loader.dataset), len(val_loader.dataset))

    # --- modele
    model = build_model(cfg).to(device)
    logger.info("Modele %s : %s parametres entrainables",
                cfg.model.name, f"{count_parameters(model):,}".replace(",", " "))

    # --- loss ponderee (compense le desequilibre des classes)
    weights = compute_class_weights(train_labels, cfg.train.class_weighting).to(device)
    logger.info("Poids de classes : %s", np.round(weights.cpu().numpy(), 3).tolist())
    criterion = nn.CrossEntropyLoss(weight=weights,
                                    label_smoothing=cfg.train.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler(enabled=cfg.train.amp and device.type == "cuda")

    # SWA : on commence a moyenner apres swa_start_frac de l'entrainement,
    # quand le taux d'apprentissage est deja bas et le modele dans un bassin stable.
    use_swa = bool(cfg.train.get("swa", False))
    swa_start = int(cfg.train.epochs * float(cfg.train.get("swa_start_frac", 0.75)))
    swa_model = AveragedModel(model) if use_swa else None
    swa_count = 0
    # swa_lr non nul : on remplace le cosinus par un taux CONSTANT pendant la
    # phase SWA. Les poids explorent alors un plateau de la surface de perte au
    # lieu de converger vers un point unique, ce que la moyenne exploite.
    # swa_lr nul : on garde le cosinus, la moyenne porte sur la queue de
    # l'entrainement (variante plus conservatrice).
    swa_lr = cfg.train.get("swa_lr", None)
    swa_scheduler = None
    if use_swa:
        if swa_lr:
            swa_scheduler = SWALR(optimizer, swa_lr=float(swa_lr), anneal_epochs=5)
        logger.info("SWA active a partir de l'epoque %d (swa_lr = %s)",
                    swa_start + 1, swa_lr if swa_lr else "cosinus conserve")

    history: Dict[str, list] = {"train_loss": [], "val_loss": [],
                                "val_acc": [], "val_macro_f1": []}
    best_f1, best_epoch, patience = -1.0, 0, 0
    start = time.time()

    for epoch in range(1, cfg.train.epochs + 1):
        # Pendant la phase SWA avec taux constant, le cosinus est desactive.
        active_scheduler = scheduler
        if use_swa and swa_scheduler is not None and epoch > swa_start:
            active_scheduler = None

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     active_scheduler, scaler, device, cfg)
        val_loss, y_true, y_pred = evaluate(model, val_loader, eval_criterion, device)
        metrics = full_report(y_true, y_pred, cfg.model.num_classes)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metrics["macro_f1"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(metrics["accuracy"])
        history["val_macro_f1"].append(metrics["macro_f1"])

        flag = ""
        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch, patience = metrics["macro_f1"], epoch, 0
            torch.save({
                "model_state": model.state_dict(),
                "config": json.loads(json.dumps(cfg)),
                "epoch": epoch,
                "val_macro_f1": best_f1,
            }, run_dir / "best.pt")
            flag = "  <- meilleur, sauvegarde"
        else:
            patience += 1

        logger.info(
            "Epoque %3d/%d | train %.4f | val %.4f | acc %.4f | macroF1 %.4f | lr %.2e%s",
            epoch, cfg.train.epochs, train_loss, val_loss, metrics["accuracy"],
            metrics["macro_f1"], optimizer.param_groups[0]["lr"], flag,
        )

        if use_swa and epoch > swa_start:
            swa_model.update_parameters(model)
            swa_count += 1
            if swa_scheduler is not None:
                swa_scheduler.step()

        if use_swa and epoch >= swa_start:
            patience = 0  # ne pas interrompre la phase de moyennage

        if patience >= cfg.train.early_stopping_patience:
            logger.info("Early stopping : aucun progres depuis %d epoques", patience)
            break

    duration = time.time() - start
    logger.info("Termine en %.1f min | meilleur macro-F1 = %.4f (epoque %d)",
                duration / 60, best_f1, best_epoch)

    # --- SWA : moyenne des poids, evaluee et comparee au meilleur checkpoint
    swa_f1 = None
    if use_swa:
        if swa_count < 2:
            logger.warning("SWA ignore : seulement %d epoques moyennees "
                           "(arret anticipe trop precoce ?)", swa_count)
        else:
            logger.info("SWA : %d epoques moyennees", swa_count)
            swa_f1, _, _, _ = finalize_swa(swa_model, train_loader, val_loader,
                                           eval_criterion, device, cfg, run_dir,
                                           best_f1, logger)

    # --- evaluation finale avec le MEILLEUR checkpoint, pas le dernier
    checkpoint = torch.load(run_dir / "best.pt", map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    _, y_true, y_pred = evaluate(model, val_loader, eval_criterion, device)
    final = full_report(y_true, y_pred, cfg.model.num_classes)
    logger.info(format_report(y_true, y_pred, cfg.model.num_classes))

    plot_history(history, run_dir / "history.png")
    plot_confusion_matrix(y_true, y_pred, run_dir / "confusion_matrix.png",
                          num_classes=cfg.model.num_classes)
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    log_experiment(cfg, final, {
        "swa_macro_f1": round(swa_f1, 4) if swa_f1 is not None else "",
        "best_epoch": best_epoch,
        "duration_min": round(duration / 60, 1),
        "n_params": count_parameters(model),
    })
    logger.info("Artefacts ecrits dans %s", run_dir)


if __name__ == "__main__":
    main()
