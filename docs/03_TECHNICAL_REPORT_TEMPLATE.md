# Technical Report — GAN Augmentation for Defect Recall in a CNN Casting Inspection System

*2000–3000 words. Replace every [ ] with numbers from your own `outputs/`; never invent numbers; delete italic instructions.*
**Student(s):** [ ]  **Course:** CSD402, CA3 Capstone, Option A  **Date:** [ ]

## Abstract (≈ 120 words)
*Problem, data, models, headline result with CI, answer to the core question, Grad-CAM finding, recommendation.* "On [N] casting images with the defect class reduced to [10 %] of the training set, ResNet-18 transfer learning reached macro-F1 [ ] versus [ ] from scratch. Adding [n] DCGAN images changed test defect recall from [x ± y] to [x ± y] (difference [d], 95 % CI [a,b]); class weighting and oversampling gave [ ] and [ ]. We conclude [ ] because [ ]. Grad-CAM shows [ ]."

## 1. Problem framing (≈ 250 words)
Domain, users, input/output, cost of missed defect vs false alarm, **why defect recall is the headline metric** (with precision/F1/PR-AUC alongside), hypothesis and success criterion from the proposal.

## 2. Data and leakage control (≈ 400 words)
Source and citation; image size/mode; class counts per split (`class_distribution.csv`); natural vs controlled imbalance and why the reduction is in train only; **near-duplicate groups** (`data_report.json`) and group-aware split; augmentation choices and domain justification (flip/rotation valid for round castings; colour jitter limited because grey-level may signal defects); normalisation with ImageNet statistics; leakage controls (GAN sees train minority only — asserted; vocabulary-free but augmentation/normalisation only train-fitted; test once, `test_access.log`).

## 3. Models (≈ 500 words)
| | Scratch CNN | ResNet-18 (pre-trained) | DCGAN |
|---|---|---|---|
| Parameters | [ ] | [ ] | G [ ], D [ ] |
| Design | 4×(Conv-BN-ReLU×2, MaxPool), GAP, dropout 0.3 | frozen backbone [3] epochs then fine-tune; dropout 0.3 head | transposed-conv G, strided-conv D, BN, ReLU/LeakyReLU, N(0,0.02) init |
| Optimiser | AdamW, wd 1e-4, cosine | same | Adam 2e-4, β1 = 0.5 |
| Selected LR | [ ] | [ ] | — |
Justify: why transfer learning for small data; skip connections; why two-phase training; GAN minimax objective, non-saturating G loss, one-sided label smoothing; experiment arms and why oversampling/GAN add equal sample counts.

## 4. Training and diagnosis (≈ 400 words)
`learning_curves.png`; per run: observation → diagnosis → remedy → effect. Compare scratch vs pre-trained (overfitting gap), baseline accuracy vs minority recall (imbalance). Validation table mean ± sd.

## 5. GAN analysis (≈ 350 words)
`gan_losses.png`, **uncurated** `gan_samples.png`, `gan_info.json` (diversity ratio). Realism, diversity, mode collapse?, discriminator dominance?; **resolution-confound** (64×64 vs original) and the `--degrade-res`/`--gan-size 128` control result.

## 6. Results on the held-out test set (≈ 450 words)
Table 1: macro-F1, defect recall, precision, F1, PR-AUC, accuracy, mean ± sd for scratch, ResNet-18 (baseline), class weights, oversampling, GAN (`test_results_mean_std.csv`). Table 2: bootstrap CI of recall difference vs baseline with baseline seed sd (`bootstrap_minority_recall_difference.json`). Table 3: threshold tuning (`threshold_tuning_target_recall.csv`). Table 4: cost. Figures: `minority_recall.png`, `confusion_matrices.png`.
**Conclusion on augmentation (two sentences, with CI) and why.**

## 7. Grad-CAM explainability (≈ 400 words)
Method (equation), figure `gradcam_<arm>.png` with ≥ 8 test examples including FN and FP; border-share table; what the model attends to (defect vs rim/background/lighting); examples of misleading maps; would an engineer trust them; limits (coarse 7×7 resolution upsampled, class-discriminative only, not causal).

## 8. Limitations and future work (≈ 250 words)
Dataset size/domain; single GAN training; 64×64 synthetic resolution; one split; 3 seeds; threshold depends on validation prevalence; lighting/camera shift; conditions under which the system must not be deployed (new casting types, changed camera, as sole safety gate). Future: conditional GAN, FID, EfficientNet-B0, patch-level/anomaly detection, more real defect images.

## 9. Conclusion (≈ 80 words)

## References and declarations
He et al. 2016 (ResNet); Radford et al. 2016 (DCGAN); Goodfellow et al. 2014 (GAN); Selvaraju et al. 2017 (Grad-CAM); dataset; libraries. **AI-assistance declaration:** [precise statement].
## Appendix — reproduction commands, software versions, GPU.
