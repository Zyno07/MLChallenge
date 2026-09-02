# Challenge ML 2026 — Classification d'affiches de films

Classification d'affiches en 5 genres : `0` Animation, `1` Blockbuster, `2` Horreur,
`3` Comédie, `4` Art & Essai.

**Contrainte du sujet : aucun modèle pré-entraîné.** Toutes les architectures de
`src/models.py` sont construites à la main et initialisées aléatoirement (Kaiming).
Aucun poids externe n'est chargé nulle part dans ce dépôt.

---

## 1. Installation

### 1.1 Prérequis

- Python 3.10 ou plus récent
- Visual Studio Code avec l'extension **Python** (Microsoft)
- Git

### 1.2 Mise en place

```bash
git clone <url-de-votre-depot> movie-poster-classifier
cd movie-poster-classifier

# environnement virtuel
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Si vous avez un GPU NVIDIA, installez PyTorch avec CUDA depuis
<https://pytorch.org/get-started/locally/> **avant** le `pip install -r`.

### 1.3 Configuration de VS Code

`Ctrl+Shift+P` → `Python: Select Interpreter` → choisissez `.venv`.

Créez `.vscode/settings.json` :

```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.analysis.extraPaths": ["."],
  "editor.rulers": [88],
  "files.trimTrailingWhitespace": true
}
```

Et `.vscode/launch.json` pour lancer un entraînement avec `F5` :

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Entrainement",
      "type": "debugpy",
      "request": "launch",
      "module": "src.train",
      "args": ["--config", "configs/default.yaml"],
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

### 1.4 Vérifier l'installation

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 2. Tester le projet sans les vraies données

À faire **immédiatement**, avant même de recevoir les images. Le script génère de
fausses affiches dont la signature colorimétrique dépend de la classe, ce qui permet
de vérifier que toute la chaîne apprend réellement.

```bash
python scripts/make_dummy_data.py --n-train 400 --n-test 80
python scripts/check_data.py --config configs/default.yaml
python -m src.train --config configs/default.yaml \
    --set run_name=smoke_test train.epochs=8 model.name=poster_cnn_small
python -m src.predict --checkpoint outputs/smoke_test/best.pt --dry-run \
    --output /tmp/dryrun.csv
```

Sur ces fausses données, le macro-F1 doit dépasser 0,90. S'il stagne vers 0,20, le
problème est dans votre code, pas dans le modèle.

Quand vous avez les vraies données, supprimez `data/images/`, `data/test/`,
`data/train_labels.csv` et remplacez-les.

---

## 3. Arborescence

```
movie-poster-classifier/
├── configs/
│   └── default.yaml           # toute la configuration, un seul endroit
├── data/
│   ├── images/                # affiches d'entraînement
│   ├── train_labels.csv       # nom_image,label  (sans en-tête)
│   ├── test/                  # affiches de test (jour J)
│   └── test_list.csv          # ordre exact des images de test
├── scripts/
│   ├── check_data.py          # contrôles avant entraînement
│   └── make_dummy_data.py     # faux jeu de données
├── src/
│   ├── utils.py               # config, seeds, device, logs
│   ├── data.py                # dataset, split, augmentation
│   ├── models.py              # architectures CNN maison
│   ├── metrics.py             # macro-F1, matrice de confusion, figures
│   ├── train.py               # boucle d'entraînement
│   ├── baselines.py           # ML classique sur descripteurs
│   ├── predict.py             # génération du CSV de soumission
│   └── ensemble.py            # moyenne de plusieurs modèles
├── outputs/                   # un sous-dossier par run
├── submissions/               # fichiers CSV à rendre
└── experiments.csv            # journal de toutes les expériences
```

---

## 4. Utilisation

### 4.1 Vérifier les données

```bash
python scripts/check_data.py --config configs/default.yaml
```

Le script affiche notamment les valeurs `DEFAULT_MEAN` et `DEFAULT_STD` calculées sur
**votre** jeu d'entraînement. Reportez-les dans `src/data.py` avant le premier vrai
entraînement.

### 4.2 Lancer les baselines

```bash
python -m src.baselines --config configs/default.yaml --models logreg rf svm
```

Les descripteurs sont mis en cache dans `outputs/baseline_features.npz` : la
deuxième exécution est instantanée.

### 4.3 Entraîner le CNN

```bash
python -m src.train --config configs/default.yaml
```

Surcharger un paramètre sans toucher au fichier de config :

```bash
python -m src.train --config configs/default.yaml \
    --set run_name=lr3e4_w48 train.lr=0.0003 model.width=48
```

Chaque run produit dans `outputs/<run_name>/` :
`best.pt`, `config.yaml`, `train.log`, `history.png`, `confusion_matrix.png`,
`history.json`. Et une ligne dans `experiments.csv`.

### 4.4 Générer la soumission

À blanc, sur la validation (à faire **avant** le jour J) :

```bash
python -m src.predict --checkpoint outputs/<run>/best.pt --dry-run \
    --output /tmp/dryrun.csv --tta
```

En réel :

```bash
python -m src.predict \
    --checkpoint outputs/<run>/best.pt \
    --test-dir data/test \
    --test-list data/test_list.csv \
    --output submissions/nom1_nom2_nom3_nom4.csv \
    --tta \
    --save-probs outputs/probs/<run>.npy
```

Le script relit son propre fichier et vérifie : nombre de lignes, nombre de colonnes,
ordre des noms, validité des labels. Il refuse d'aboutir si quelque chose cloche.

### 4.5 Ensemble

```bash
python -m src.ensemble \
    --probs outputs/probs/a.npy outputs/probs/b.npy outputs/probs/c.npy \
    --test-list data/test_list.csv \
    --output submissions/nom1_nom2_nom3_nom4.csv
```

---

## 5. Modifier le projet

### Changer d'architecture

Dans `configs/default.yaml` : `model.name` = `poster_cnn`, `poster_cnn_small` ou
`poster_resnet`. Pour en ajouter une, écrivez la classe dans `src/models.py`,
respectez la signature `(num_classes, width, num_blocks, dropout)`, puis
enregistrez-la dans le dictionnaire `MODELS`.

### Changer l'augmentation

Section `augment` du YAML. Attention à `hue` : la teinte est un signal très
discriminant ici (horreur sombre et désaturée, animation saturée). Un jitter fort
détruit ce signal. Testez `hue=0.0` contre `hue=0.03` contre `hue=0.10` et notez.

### Gérer le déséquilibre

Deux leviers, à ne pas cumuler :

- `train.class_weighting: balanced` — pondère la loss
- `train.sampler: weighted` — rééchantillonne (mettre alors `class_weighting: none`)

Comparez les deux, c'est un bon paragraphe de rapport.

### Ajouter une métrique

`src/metrics.py`, puis référencez-la dans `full_report()`. Elle apparaîtra
automatiquement dans `experiments.csv`.

---

## 6. Protocole expérimental

1. Le split de validation est figé par `seed`. **Ne le changez jamais** en cours de
   projet, sinon vos résultats ne sont plus comparables entre eux.
2. Une expérience = **une seule variable modifiée**. Sinon vous ne saurez pas ce qui
   a produit le gain.
3. `experiments.csv` se remplit tout seul. C'est la matière brute de votre partie
   « Résultats ». Ne le reconstituez pas de mémoire à la fin.
4. La sélection du modèle se fait **exclusivement** sur le macro-F1 de validation.
   Le jeu de test ne sert jamais à choisir quoi que ce soit.

### Ordre de priorité des expériences

| Priorité | Expérience | Gain attendu |
|---|---|---|
| 1 | Augmentation (flip, affine, jitter) | +5 à 12 pts |
| 2 | Pondération des classes | +4 à 8 pts macro-F1 |
| 3 | Taille d'image 160×240 → 192×288 | +2 à 5 pts |
| 4 | Profondeur / largeur du réseau | +2 à 5 pts |
| 5 | Ensemble de 3 seeds + TTA | +2 à 4 pts |
| 6 | Mixup, label smoothing | +1 à 3 pts |
| 7 | Learning rate, weight decay | +1 à 2 pts |

Faites-les dans cet ordre. Les hyperparamètres en dernier, ils rapportent le moins.

---

## 7. Entraînement sur Google Colab

Sur CPU, une époque prend plusieurs minutes. Sur GPU Colab (gratuit), quelques
secondes. Créez un notebook :

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone <url-du-depot> /content/projet
%cd /content/projet
!pip install -q -r requirements.txt

# les données restent sur Drive, on les lie
!ln -s /content/drive/MyDrive/challenge_ml/data /content/projet/data

!python -m src.train --config configs/default.yaml --set run_name=colab_run

# IMPORTANT : recopier les résultats sur Drive avant déconnexion
!cp -r outputs /content/drive/MyDrive/challenge_ml/
```

Vérifiez `Exécution → Modifier le type d'exécution → GPU T4`.

---

## 8. Pièges à éviter

- **`shuffle=True` sur le loader de test.** Vos prédictions seraient dans le
  désordre et le fichier invalide. `build_test_loader` force `shuffle=False`.
- **Normalisation calculée sur train+val.** C'est une fuite de données.
  `check_data.py` ne calcule les statistiques que sur le train.
- **Augmentation appliquée en validation.** Vos scores deviennent bruités et
  incomparables. Séparez bien `build_train_transform` et `build_eval_transform`.
- **Rapporter l'accuracy seule.** Avec 4,5 % de Blockbuster, elle ment.
- **Évaluer avec le dernier checkpoint** au lieu du meilleur. `train.py` recharge
  `best.pt` avant l'évaluation finale.
- **Perdre les poids** à la déconnexion de Colab. Copiez sur Drive après chaque run.
- **Écrire le rapport le dernier jour.** Remplissez-le au fil des expériences.