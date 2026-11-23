# Capstone Project Proposal — CSD402 Deep Learning

**Title:** Does GAN-generated data improve defect recall? A CNN + DCGAN visual inspection system for casting defects
**Student(s):** [name(s), roll no.]  **Option:** A — Multi-Stage Visual Quality System (CNN + GAN)

## 1. Problem statement
In casting production, defective parts are rare, so a classifier trained on production data sees few defects and tends to miss them. **Task:** input = casting image; output = {ok, defect}; a missed defect (false negative) is far more costly than a false alarm, so the **primary metric is recall on the defect class**, reported with precision, F1, PR-AUC and macro-F1 (never accuracy alone).

## 2. Core question and hypothesis
*Does GAN-generated augmentation of the under-represented defect class improve defect recall on a real, held-out test set the GAN never saw, compared with a non-augmented baseline and with simple remedies (class weights, oversampling)?*
**Hypothesis:** GAN augmentation raises test defect recall by at least [5] points over the baseline (95 % bootstrap CI excluding 0, larger than the seed sd) without precision falling below [X]. A null or negative result will be diagnosed and reported.

## 3. Dataset
Kaggle casting-product defect images (simmoshaikh/casting-defect-detection) [state the real number of images and classes]. Near-duplicate-safe stratified split (≈ 71/14/14). The defect class is **deliberately reduced to [10 %] in the training split only**; validation/test keep the natural distribution.

## 4. Planned architectures
* **Stage 1:** a 4-block CNN from scratch vs **ResNet-18 pre-trained on ImageNet** (two-phase transfer learning: frozen backbone, then fine-tuning). Augmentation (random resized crop, flip, rotation, colour jitter), dropout 0.3, AdamW + cosine schedule, early stopping on validation macro-F1.
* **Stage 2:** **DCGAN** (64×64) trained only on training-split defect images; synthetic images added to the training split only, in the same number as random oversampling (matched class balance).
* **Arms:** baseline, class-weighted loss, random oversampling, GAN augmentation; identical hyper-parameters, 3 seeds.
Justification: ResNet-18 trains well on small data through transfer learning and skip connections; DCGAN is the standard stable convolutional GAN.

## 5. Evaluation plan
Selection on validation only; test evaluated once. Metrics: defect recall/precision/F1/PR-AUC, macro-F1, confusion matrices; mean ± sd over 3 seeds; paired bootstrap CI of recall differences; threshold tuning on validation for ≥ 95 % recall. GAN diagnostics: loss curves, uncurated samples, diversity ratio, resolution-confound control.

## 6. Unit VI component
**Grad-CAM** on ≥ 8 test images (TP, FN, FP, TN) with a border-share check for spurious cues, and a discussion of whether a domain expert would trust the maps.

## 7. Risks and timeline
Risks: GAN mode collapse or poor realism; tiny minority set; compute. Mitigation: smaller images, fewer epochs, Colab GPU. Timeline: W3 data + baseline, W4 CNN comparison, W5 GAN + test + Grad-CAM, W6 report + presentation.
