
# electronic-component-detection-dl

Image classification of 21 electronic component types with PyTorch + transfer learning.

**Dataset:** [Electronic Components (Kaggle, aryaminus)](https://www.kaggle.com/datasets/aryaminus/electronic-components), with some images manually removed during cleaning.

## Problem setup

- **21 classes** (armature, attenuator, capacitors, cartridge-fuse, filament, heat-sink, induction-coil, integrated-circuits, interconnects, LED, light-circuit, limiter-clipper, local-oscillator, omni-directional-antenna, potentiometers, pulse-generator, relays, semiconductor-diode, shunt, transformers, transistors).
- **~10.5K images total**, 80/20 stratified train/val split.
- **Severe class imbalance** (~64x ratio): `integrated-circuits` has 1725 samples, `armature` has 27.

| Class | Count |   | Class | Count |
|---|---:|---|---|---:|
| integrated-circuits | 1725 | | filament | 400 |
| transistors | 1372 | | pulse-generator | 310 |
| relays | 1179 | | attenuator | 248 |
| potentiometers | 875 | | cartridge-fuse | 225 |
| transformers | 725 | | induction-coil | 151 |
| capacitors | 705 | | omni-directional-antenna | 118 |
| interconnects | 520 | | local-oscillator | 112 |
| LED | 459 | | shunt | 76 |
| limiter-clipper | 427 | | light-circuit | 57 |
| semiconductor-diode | 417 | | armature | 27 |
| heat-sink | 406 | | | |

## Pipeline

- Trained in Google Colab on a T4 GPU.
- Dataset copied from Google Drive to local Colab disk before training (~20x I/O speedup vs. reading through the Drive FUSE mount).
- ImageFolder + stratified split (random_state=42) for reproducibility.
- ImageNet normalization, resize to 224x224.
- Mixed augmentation pipeline (later trials): RandomAffine, ColorJitter, RandomHorizontalFlip, TrivialAugmentWide, RandomErasing.
- Adam optimizer with discriminative learning rates per layer group.
- ReduceLROnPlateau scheduler on val_loss.
- Early stopping (patience=3, min_delta=0.01) + best-checkpoint snapshotting.

## Trials and results

All results are on the held-out 20% validation split. "Plain acc" is overall accuracy; "Macro F1" averages F1 across all classes equally and is the honest metric for an imbalanced dataset.

| # | Backbone | Key changes | Plain acc | Macro F1 | Notes |
|---|----------|-------------|-----------|----------|-------|
| 1 | ResNet18 | Frozen except `layer4`; dropout 0.3; basic aug; Adam lr=1e-4 | ~0.663 | n/a | **Overfitting**: train 83% / val 66% at epoch 8; val_loss rising after epoch 5. |
| 2 | ResNet18 | Unfroze `layer3`+`layer4` with discriminative LRs (1e-5 / 5e-5 / 1e-4); dropout 0.5; weight_decay 1e-4; stronger aug (TrivialAugmentWide, RandomErasing); ReduceLROnPlateau; best-state snapshotting | **0.674** | n/a | Train/val gap closed (68% / 67%). Best at epoch 15, val_loss 1.5062. |
| 3 | EfficientNet-B0 | Same recipe as trial 2; unfroze `features[6]`+`features[7]`; new head (Dropout 0.4 → 512 → Dropout 0.5 → 21) | ~0.667 | n/a | ~32 epochs (continued past the 20-epoch cap). Val_loss 1.4671 — best loss seen, but accuracy did not beat ResNet18. |
| 4 | EfficientNet-B0 | + WeightedRandomSampler with weight = 1/n (aggressive class rebalancing) | 0.610 | 0.567 | Plain acc drops, but minority classes finally visible (omni-directional-antenna F1 0.89, pulse-generator 0.73). Majority recall crashed (potentiometers 0.39, relays 0.43). |
| 5 | EfficientNet-B0 | + WeightedRandomSampler with weight = 1/sqrt(n) (softer rebalancing) | 0.634 | **0.589** | Best balance overall. Majority recall recovered (potentiometers 0.53, relays 0.49); minority gains preserved (cartridge-fuse F1 0.85, shunt 0.69, filament 0.82). |

### Trial 5 per-class breakdown (best run)

Strong (F1 > 0.70): `interconnects` (0.84), `cartridge-fuse` (0.85), `filament` (0.82), `integrated-circuits` (0.73), `pulse-generator` (0.73), `omni-directional-antenna` (0.82).

Mid (F1 0.50–0.70): `LED`, `attenuator`, `limiter-clipper`, `potentiometers`, `relays`, `transistors`, `transformers`, `shunt`, `heat-sink`.

Weak (F1 < 0.50): `capacitors` (0.47), `induction-coil` (0.45), `semiconductor-diode` (0.42).

**Effectively broken** (data problem, not tuning problem):
- `light-circuit` — F1 0.00 (only 57 train samples, high visual variance).
- `local-oscillator` — F1 0.12 (112 train samples, visually overlaps with other PCB-style classes).

## Findings

1. **The I/O bottleneck was Google Drive, not the GPU.** Copying the dataset to local Colab disk and bumping `num_workers` from 0 to 2 with `persistent_workers=True` cut epoch time from minutes to seconds.
2. **Regularization closed the train/val gap** — trial 1 had a 17-point train/val gap; trial 2's recipe (heavier aug + dropout 0.5 + weight_decay + discriminative LR + ReduceLROnPlateau) brought it to under 1 point.
3. **Bigger backbone did not help.** EfficientNet-B0 (trial 3) matched but did not beat ResNet18 (trial 2). With ~10K images across 21 classes, the dataset — not model capacity — was the bottleneck.
4. **Plain accuracy is misleading on imbalanced data.** Pre-rebalancing accuracy of 67% mostly reflected good performance on the top 5 classes. Macro-F1 on minority classes was near zero. Rebalancing trades a few points of plain accuracy for substantially better minority-class performance.
5. **`1/sqrt(n)` sample weights beat `1/n`.** Fully balanced sampling starved the majority classes; the square-root softening kept minority gains while letting the model still see common classes frequently.
6. **Two classes are unfixable without more data / label review** — `light-circuit` and `local-oscillator`. The headline number would jump if these were merged with a related class or dropped.

## Possible next steps (not pursued)

- Investigate `light-circuit` and `local-oscillator`: review labels, consider merging into a `miscellaneous-circuit` class or dropping.
- Collect more samples for the bottom 6 classes (each <150 images).
