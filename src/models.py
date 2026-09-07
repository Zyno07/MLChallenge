"""Architectures CNN construites entierement a la main.

AUCUN poids pre-entraine n'est charge ici : toutes les couches sont
initialisees aleatoirement (Kaiming) et entrainees depuis zero, conformement
a la contrainte du sujet.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------- briques
class ConvBlock(nn.Module):
    """[Conv 3x3 -> BatchNorm -> ReLU] x2 -> MaxPool 2x2.

    BatchNorm est indispensable ici : sans elle, un reseau profond entraine
    depuis zero sur ~3000 images converge tres mal.
    """

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return self.pool(x)


class ResidualBlock(nn.Module):
    """Bloc residuel maison (architecture inspiree de ResNet, poids aleatoires)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


# --------------------------------------------------------------- modeles
class PosterCNN(nn.Module):
    """CNN principal : N blocs conv, GlobalAveragePooling, classifieur.

    GlobalAveragePooling plutot que Flatten + Dense : divise le nombre de
    parametres par ~50 et reduit fortement le surapprentissage, ce qui est
    critique avec seulement ~3000 images d'entrainement.
    """

    def __init__(self, num_classes: int = 5, width: int = 32,
                 num_blocks: int = 4, dropout: float = 0.4) -> None:
        super().__init__()
        channels = [3] + [width * (2 ** i) for i in range(num_blocks)]
        self.features = nn.Sequential(*[
            ConvBlock(channels[i], channels[i + 1]) for i in range(num_blocks)
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)

    @torch.no_grad()
    def feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Cartes d'activation du dernier bloc (utile pour Grad-CAM / rapport)."""
        return self.features(x)


class PosterCNNSmall(nn.Module):
    """Version legere : entrainement rapide sur CPU, sert de test de pipeline."""

    def __init__(self, num_classes: int = 5, width: int = 16,
                 num_blocks: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        channels = [3] + [width * (2 ** i) for i in range(num_blocks)]
        layers: list[nn.Module] = []
        for i in range(num_blocks):
            layers += [
                nn.Conv2d(channels[i], channels[i + 1], 3, padding=1, bias=False),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels[-1], num_classes)
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)))


class PosterResNet(nn.Module):
    """ResNet maison (poids aleatoires). Plus profond, meilleur si assez d'epoques."""

    def __init__(self, num_classes: int = 5, width: int = 32,
                 num_blocks: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        blocks: list[nn.Module] = []
        in_ch = width
        for i in range(num_blocks):
            out_ch = width * (2 ** i)
            stride = 1 if i == 0 else 2
            blocks.append(ResidualBlock(in_ch, out_ch, stride=stride))
            blocks.append(ResidualBlock(out_ch, out_ch))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, num_classes)
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(self.stem(x))))


class PosterResNetV2(nn.Module):
    """ResNet maison a sous-echantillonnage reduit.

    MOTIVATION
    ----------
    PosterResNet divise la resolution par 32 avant le classifieur : stride 2
    dans le stem, MaxPool, puis trois blocs de stride 2. Sur une affiche de
    288 pixels de haut, il ne reste que 9 pixels, contre 18 pour PosterCNN.
    Le detail de texture qui distingue un rendu dessine d'une photographie est
    perdu, ce qui explique son macro-F1 hors-pli de 0,628 contre 0,674.

    Cette variante retire le MaxPool du stem : facteur 16 au lieu de 32, soit
    la meme resolution finale que PosterCNN, tout en conservant les connexions
    residuelles. L'objectif est un membre d'ensemble a la fois PERFORMANT et
    DECORRELE des CNN simples, la decorrelation etant ce qui fait la valeur
    d'une agregation.

    PosterResNet est conserve intact : ses checkpoints restent chargeables.
    """

    def __init__(self, num_classes: int = 5, width: int = 48,
                 num_blocks: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        # Facteur 2 seulement (pas de MaxPool), contre 4 pour PosterResNet.
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        blocks: list[nn.Module] = []
        in_ch = width
        for i in range(num_blocks):
            out_ch = width * (2 ** i)
            stride = 1 if i == 0 else 2
            blocks.append(ResidualBlock(in_ch, out_ch, stride=stride))
            blocks.append(ResidualBlock(out_ch, out_ch))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, num_classes)
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(self.stem(x))))


class GramLayer(nn.Module):
    """Matrice de Gram : correlations entre canaux d'une carte d'activation.

    PRINCIPE
    --------
    Pour une carte F de forme (C, H, W), la matrice de Gram vaut
    G = F F^T / (H W), de taille (C, C). Chaque coefficient G_ij mesure a quel
    point les filtres i et j s'activent conjointement, INDEPENDAMMENT de la
    position spatiale. On capture donc le STYLE (textures, palettes,
    contrastes) plutot que le CONTENU (quel objet, ou).

    MOTIVATION POUR CE PROJET
    -------------------------
    Notre hypothese initiale etait que les genres possedent une signature
    stylistique : horreur sombre et desaturee, animation claire et saturee.
    C'est precisement ce que mesure une matrice de Gram. Wi et al. (2020)
    rapportent un gain de 1 a 2 points avec ce mecanisme sur la meme tache.

    Une reduction 1x1 precede le calcul : avec C = 384 canaux, la matrice
    ferait 384x384 = 147 456 coefficients, ingerable. On reduit a `gram_dim`
    canaux, soit gram_dim(gram_dim+1)/2 coefficients apres extraction du
    triangle superieur (la matrice est symetrique).
    """

    def __init__(self, in_channels: int, gram_dim: int = 64) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, gram_dim, 1, bias=False),
            nn.BatchNorm2d(gram_dim),
            nn.ReLU(inplace=True),
        )
        self.gram_dim = gram_dim
        idx = torch.triu_indices(gram_dim, gram_dim)
        self.register_buffer("triu_idx", idx, persistent=False)
        self.out_features = gram_dim * (gram_dim + 1) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        b, c, h, w = x.shape
        f = x.reshape(b, c, h * w)
        gram = torch.bmm(f, f.transpose(1, 2)) / (h * w)
        gram = gram[:, self.triu_idx[0], self.triu_idx[1]]
        # Racine signee puis normalisation L2 : les coefficients de Gram ont
        # une dynamique tres large, sans quoi quelques dimensions ecrasent
        # les autres et le classifieur ne converge pas.
        gram = torch.sign(gram) * torch.sqrt(gram.abs() + 1e-6)
        return nn.functional.normalize(gram, dim=1)


class PosterCNNGram(nn.Module):
    """PosterCNN augmente d'une branche de style (matrice de Gram).

    Le classifieur recoit la concatenation de deux representations :
      - contenu : global average pooling du dernier bloc, comme PosterCNN ;
      - style   : matrice de Gram calculee sur l'avant-dernier bloc.

    L'avant-dernier bloc est choisi car les couches intermediaires portent
    davantage d'information de texture, les dernieres etant plus semantiques.
    """

    def __init__(self, num_classes: int = 5, width: int = 48,
                 num_blocks: int = 4, dropout: float = 0.3,
                 gram_dim: int = 64) -> None:
        super().__init__()
        channels = [3] + [width * (2 ** i) for i in range(num_blocks)]
        self.blocks = nn.ModuleList([
            ConvBlock(channels[i], channels[i + 1]) for i in range(num_blocks)
        ])
        self.gram_from = max(0, num_blocks - 2)   # avant-dernier bloc
        self.gram = GramLayer(channels[self.gram_from + 1], gram_dim)

        self.pool = nn.AdaptiveAvgPool2d(1)
        fused = channels[-1] + self.gram.out_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        style = None
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == self.gram_from:
                style = self.gram(x)
        content = self.pool(x).flatten(1)
        return self.head(torch.cat([content, style], dim=1))


# ------------------------------------------------------------ initialisation
def init_weights(module: nn.Module) -> None:
    """Initialisation Kaiming : adaptee aux activations ReLU."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0, 0.01)
        nn.init.zeros_(module.bias)


# ------------------------------------------------------------------ factory
MODELS = {
    "poster_cnn": PosterCNN,
    "poster_cnn_small": PosterCNNSmall,
    "poster_resnet": PosterResNet,
    "poster_resnet_v2": PosterResNetV2,
    "poster_cnn_gram": PosterCNNGram,
}


def build_model(cfg) -> nn.Module:
    name = cfg.model.name
    if name not in MODELS:
        raise ValueError(f"Modele inconnu : {name}. Choix : {list(MODELS)}")
    kwargs = dict(
        num_classes=cfg.model.num_classes,
        width=cfg.model.width,
        num_blocks=cfg.model.num_blocks,
        dropout=cfg.model.dropout,
    )
    if name == "poster_cnn_gram":
        kwargs["gram_dim"] = int(cfg.model.get("gram_dim", 64))
    return MODELS[name](**kwargs)
