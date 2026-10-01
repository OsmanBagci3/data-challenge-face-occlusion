# Face Occlusion Prediction

Prédire la **proportion de pixels occultés** d'un visage, ŷ ∈ [0, 1], à partir d'une image recadrée 224×224 — sous une métrique **pondérée et sensible au genre**.

> DATA704 — Machine Learning Avancé · Télécom Paris / IP Paris · 2025–2026
> Groupe 5 *DeepVision* — encadrement : Charlotte Laclau, Stephan Alaniz

| | |
|---|---|
| **Tâche** | Régression sur image, cible continue ∈ [0, 1] |
| **Données** | 100 000 crops de visages 224×224, 3 bases, 68 % hommes / 32 % femmes |
| **Score final** | **CV 0.000888** · **Leaderboard 0.00097** |
| **Baseline** | 0.00122 (MobileNet-v3-small) → **−27 %** |
| **Matériel** | 1× NVIDIA GeForce RTX 3090 |

---

## La métrique dicte tout




$Err_g = \frac{Σ_{i∈g} w_i (ŷ_i − y_i)²}{Σ_{i∈g} w_i }$,  avec $ w_i = \frac{1}{30} + y_i$

$Score = \frac{(Err_F + Err_M)}{2}    +  |Err_F − Err_M|$


Deux conséquences ont guidé chaque décision du projet :

1. **Pondération par l'occlusion réelle.** $w_i$ va de 0.033 (visage non occulté) à 1.033 (entièrement occulté) : un visage très occulté pèse **jusqu'à 30×** plus qu'un visage dégagé. Or ces cas sont rares — la distribution est fortement asymétrique à droite (médiane ≈ 5 %). Le score est dominé par une minorité d'exemples.

2. **Le gap est compté deux fois.** En notant `a` la plus grande des deux erreurs et `b` la plus petite, le score se réduit à `(3a − b)/2`. **Réduire l'écart entre genres vaut donc arithmétiquement autant que réduire l'erreur moyenne.** Dans nos expériences c'est systématiquement `Err_M > Err_F` : ce sont les hommes qui portent l'erreur dominante.

Cette asymétrie est structurelle et vient des données : 63 % des visages masculins sont dans le bucket 0–5 % d'occlusion, alors que les visages féminins sont davantage étalés vers les occlusions fortes (33 % en 10–20 %, 6.6 % en 30–50 %).

La validation (10 %) est stratifiée par **genre × bucket d'occlusion × base**, pour que le score de validation soit un proxy non biaisé du leaderboard sur toutes les strates.

## Progression

| Phase | Méthode | CV | err_F | err_M |
|---|---|---|---|---|
| Baseline | MobileNet-v3-small | 0.00122 | 0.00064 | 0.00103 |
| A | Ensemble (EffNet-B0 + ConvNeXt-Tiny) + TTA | 0.001029 | 0.00043 | 0.00083 |
| B | Ensemble pondéré 4 modèles + TTA | 0.001032 | 0.00043 | 0.00083 |
| B+ | 6 modèles + stacker LightGBM | 0.001000 | 0.00041 | 0.00081 |
| C | 7 modèles (+ MAE, DINOv2, FaceNet) + blend | 0.000983 | 0.00041 | 0.00079 |
| **E** | **Stacker MLP 5-fold × 5-seed** | **0.000888** | **0.00044** | **0.00074** |

*CV = validation croisée 5-fold honnête sur le set de validation. Soumission finale (phase E) : **0.00097** au leaderboard.*

### Les quatre leviers qui ont payé

1. **Capacité du backbone** — MobileNet → ConvNeXt-Tiny (−5 % de score).
2. **Diversité architecturale** — mélanger CNN, transformers et un modèle pré-entraîné sur des visages. Deux seeds du même ConvNeXt n'apportent rien ; Swin-Tiny et DINOv2 apportent une décorrélation réelle.
3. **Stacker non linéaire conscient du genre** — remplacer la moyenne pondérée fixe par un modèle qui peut choisir des poids *conditionnels au genre*.
4. **Loss du stacker = MSE pondérée**, et non le score complet — pour éviter de surajuster le déséquilibre de genre de la validation.

### Ce qui n'a pas marché

- **Early stopping, WeightedRandomSampler, augmentations modérées** sur la baseline : les trois échouent. Le sampler introduit une *double pondération* (la loss pondère déjà par `w_i`, d'où un gradient effectif en `w_i²`) qui dégrade `err_F`. Le goulot d'étranglement était la **capacité du modèle**, pas la régularisation.
- **Calibration isotonique par genre** (jusqu'à −9 %) : le gap s'est révélé être de la **variance résiduelle**, pas un biais monotone — une correction monotone ne peut rien y faire.
- **Deux seeds de la même architecture** : diversité négligeable pour le stacking.
- **EfficientNet-B3** : n'a pas convergé (0.00140), abandonné.

## Structure

```
├── 01_EDA.ipynb                        Exploration, distributions genre × occlusion
├── 02_split_and_baseline.ipynb         Split stratifié + baseline MobileNet
├── 03_sampler_and_aug.ipynb            Ablations : sampler, augmentations
├── 04_stronger_backbone.ipynb          Backbones timm, init du biais du head
├── 05_tta_ensemble_calibration.ipynb   TTA, calibration isotonique, ensemble
├── 06_phaseB_more_models.ipynb         Élargissement du pool de modèles
├── 07_phaseBplus_diversity_stacking.ipynb  Diversité + stacker LightGBM
├── 08_phaseC_sota_models.ipynb         MAE ViT-B, DINOv2 ViT-B, FaceNet
├── 09_phaseD_inference.ipynb           Inférence test du pool
├── 10_phaseE_mlp_inference.ipynb       Stacker MLP — système final
│
├── src/dc_utils/        Métrique officielle, I/O, split, boucle d'entraînement
├── scripts/             Entraînements longs hors notebook, benchmarks dataloader
└── experiments/         Historiques d'entraînement par modèle
```

Les répertoires `data/`, `artifacts/` et `submissions/` sont exclus du dépôt : ils
contiennent les données du challenge, les poids entraînés ou des prédictions qui en
dérivent. Ils sont recréés localement par les notebooks.

Le module `src/dc_utils/metric.py` contient l'implémentation de référence du score, en NumPy pour l'évaluation et en PyTorch (différentiable) pour servir de loss.

## Reproduction

```bash
pip install -r requirements.txt
```

**Les données ne sont pas incluses dans ce dépôt.** 

## Détail du système final

Un MLP à deux couches cachées (9 features → 64 → 64 → 1, sortie sigmoïde, biais initialisé à $log(\frac{p₀}{(1−p₀)})$ avec p₀ la moyenne d'entraînement, pour éviter la saturation), entraîné sur la MSE pondérée selon un schéma 5-fold × 5-seed, soit 25 modèles.


## Auteurs

SOP Leonnel Romuald · DJOUNANG NANA Guy Rostan · BAGCI Osman
