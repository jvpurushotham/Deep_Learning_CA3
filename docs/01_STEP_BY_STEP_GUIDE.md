# Step-by-step guide — CSD402 Capstone, Option A (CNN + GAN)

> **Start with `docs/00_LOCAL_SETUP_GUIDE.md`** to install and run everything on your computer. This guide explains the *thinking* behind each step. Dataset: only `data/raw/defect_dataset/` (ok + 11 defect types merged into "defect"); never `hand_overlays` or `visualizations`.

## 0. Deliverables (from the brief)
| # | Deliverable | Where |
|---|---|---|
| 1 | Proposal (1–2 pages, approved at Week 2) | `docs/02_PROPOSAL_DRAFT.md` / `.docx` |
| 2 | Codebase + README that reproduces all results | this repo |
| 3 | Technical report 2000–3000 words | `docs/03_TECHNICAL_REPORT_TEMPLATE.md` / `.docx` |
| 4 | Trained model artifact + loading instructions | `python run.py export` → `artifacts/model` |
| 5 | 10-minute presentation + Q&A, incl. one thing that did NOT work | `docs/04_PRESENTATION_10MIN.md` |
| 6 | Citations of weights/datasets/code/AI help | README + report |

**Marks.** The brief: 100 marks (framing & data 15, model design 25, training & tuning 20, evaluation & analysis 20, documentation & defence 20). Your 13 Nov notes show a table with **20 for coding + documentation + presentation and 10 for viva** — confirm the real weighting with your instructor.

## 1. The idea in one paragraph
A factory camera classifies castings as **ok** or **defect**. Defects are rare in real production, so a classifier sees few of them and **misses defects** (low recall on the rare class). Stage 1 builds the CNN (scratch vs pre-trained ResNet-18). Stage 2 asks whether a **GAN** — your notes' "police and thief": the *generator* (thief) forges defect images, the *discriminator* (police) tries to catch the forgeries, and both improve — can create extra defect images that **raise defect recall on real, untouched test images** compared with (i) no remedy, (ii) class-weighted loss and (iii) simple oversampling. *A negative result, properly diagnosed, is an acceptable finding* (the template says so explicitly).

## 2. CRISP-DM mapping (your notes)
| Phase | Work | Evidence |
|---|---|---|
| Business understanding | Real problem as a DL task: input = casting image, output = ok/defect, **success metric = defect recall** at acceptable precision (a missed defect is costlier than a false alarm) | proposal, report §1 |
| Data understanding | classes, image size, near-duplicates, natural imbalance | `inspect`, `data_report.json` |
| Data preparation | near-duplicate-safe stratified split 71/14/14, controlled minority reduction in train only, augmentation | `prepare` |
| Modelling | scratch CNN vs ResNet-18; DCGAN; remedies | `tune`, `gan`, `arms` |
| Evaluation | test once; minority recall/precision/F1/PR-AUC; bootstrap; Grad-CAM | `test`, `gradcam` |
| Deployment | export, threshold for target recall, limits | `export`, `threshold_tuning_target_recall.csv` |

Your notes' four reminders: (1) real problem + input/output/metric → proposal; (2) justify architecture — *"ResNet-18 because it can be trained on fewer data"*: pre-trained features (edges, textures) transfer, 11 M parameters, residual connections ease optimisation; (3) proper train/val/test → `prepare` + `test` once; (4) dropout, augmentation, overfitting curves, class imbalance → dropout 0.3, augmentations, `learning_curves.png`, class weights/oversampling/GAN.

## 3. The work, step by step
**Step 1 — Setup (30 min).** Run `python tests/smoke_test.py` (no data needed). Get the Kaggle dataset (`data/README.md`). Use a GPU (Colab).

**Step 2 — Proposal.** State the hypothesis with numbers, e.g. "GAN augmentation raises test defect recall by ≥ 5 points over the baseline with a 95 % CI excluding 0, without precision dropping below X." Without a falsifiable criterion graders see a demo, not an experiment.

**Step 3 — Data.** `inspect` → `prepare --minority defect --minority-keep 0.10`. Read `data_report.json`: natural class ratio, near-duplicate groups, final train ratio. Record the minority count in train (the GAN's training set). Think: why reduce the minority only in train? (Controlled imbalance; val/test keep the natural mix, so evaluation is realistic.)

**Step 4 — Stage 1: scratch vs pre-trained.** `python run.py tune` — 3 LRs per architecture on seed 0, validation macro-F1. Note the winner and whether it sits at a grid edge. Write the justification for choosing the final backbone from numbers, not from habit.

**Step 5 — Stage 2: GAN.** `python run.py gan`. Open `figures/gan_samples.png` (uncurated!) and `gan_losses.png`, and `gan/gan_info.json`:
* Do fakes look like real defects, or only like "casting-like blobs"?
* **Mode collapse:** all fakes look alike; `diversity_ratio_fake_over_real` ≪ 1.
* **Discriminator wins:** D loss → 0 while G loss climbs.
* Remedies you can try and document: fewer epochs, `--gan-size 128`, lower D learning rate, more augmentation of the real images (edit `capstone_a/gan.py`).
**Do not cherry-pick** samples for the report; show the uncurated grid.

**Step 6 — Arms over 3 seeds.** `python run.py arms --seeds 0 1 2` trains scratch, ResNet-18, class-weights, oversampling and GAN-augmented. They differ only in the training data/loss. Compare minority recall, precision, F1, PR-AUC on validation (mean ± sd). A difference smaller than the seed sd is not evidence.

**Step 7 — Diagnose training** (rubric: Training & Tuning 20 %). From `learning_curves.png` write for each run: observation → diagnosis (overfitting / underfitting / instability / imbalance) → remedy → effect. Useful comparisons: scratch overfits more than ResNet-18; baseline has high accuracy but low minority recall (imbalance symptom).

**Step 8 — Check for the GAN shortcut (important).** GAN images are 64×64 (blurry) while real ones are sharp. A classifier can learn "blurry ⇒ defect" and look good on training data but not improve real test recall. Control: re-run `prepare … --degrade-res 64` (all images through the same resolution) or `gan --gan-size 128`, and say what happened. This is a strong "what did not work and how I diagnosed it" story if it occurs.

**Step 9 — Test once.** `python run.py test`. Read `test_results_mean_std.csv`, `bootstrap_minority_recall_difference.json` (does the CI exclude 0? compare with the baseline seed sd), `threshold_tuning_target_recall.csv` (recall at a validation-chosen threshold: threshold tuning is a cheaper alternative to a GAN!), `cost_comparison.csv`. Never change the design after seeing test numbers.

**Step 10 — Grad-CAM (Unit VI).** `python run.py gradcam --arm gan`. Look at TP / FN / FP / TN panels. Questions: does the heat-map sit on the visible defect or on the rim/background/lighting? `cam_border_share` is the fraction of CAM energy in the outer 15 % margin (a uniform map would give ≈ 0.51) — values near or above that for confident predictions suggest a spurious cue. Would a quality engineer trust these maps? Why do missed defects (FN) show diffuse heat?

**Step 11 — Conclusion on augmentation.** Answer the core question in two sentences with numbers and CI, then diagnose *why* (e.g. synthetic images add no new information beyond the few real ones; resolution mismatch; oversampling already gives balance; or it genuinely helped because of diversity).

**Step 12 — Export + report + presentation.** `python run.py export --arm <final arm>`; `predict` sanity check; fill the report template; rehearse `docs/04_PRESENTATION_10MIN.md`; study `viva/VIVA_PREP.md`.

## 4. Schedules
**Official:** W1 idea → W2 proposal gate → W3 data pipeline + baseline → W4 core model on validation → W5 final model, test, Unit VI draft → W6 report, code, artifact, presentation.
**Crash plan (deadline unknown to me):** Day 1 smoke test, data, proposal · Day 2 `tune` + GAN · Day 3 `arms` + diagnosis · Day 4 `test`, Grad-CAM, shortcut control · Day 5 report + export · Day 6 slides + viva Parts 1–3 · Day 7 viva Parts 4–7 + mock Q&A. If compute is short: `--img-size 128`, `--epochs 10`, `--gan-epochs 150`, 2 seeds — and say so.

## 5. Rubric self-check
| Component | Evidence |
|---|---|
| Framing & data (15) | proposal; data report; near-duplicate-safe split; minority reduced in train only; GAN sees train minority only |
| Model design (25) | scratch vs ResNet-18 justified with numbers; weighted CE / oversampling / GAN compared; metric = minority recall + macro-F1 + PR-AUC, not accuracy |
| Training & tuning (20) | curves + named diagnosis + remedy; 3-LR search per backbone; 3 seeds |
| Evaluation & analysis (20) | mean ± sd; paired bootstrap CI; threshold tuning; confusion matrices; Grad-CAM with errors |
| Documentation & defence (20) | report; README reproduces; presentation with failure story; you can explain every line |

## 6. Mistakes that drop you to "Marginal"
Near-duplicates across splits · GAN trained on val/test images · choosing epochs by test score · accuracy-only on imbalanced data · tuning the GAN arm more than the baseline · one seed and a 1-point "improvement" · cherry-picked GAN samples · Grad-CAM only on correct examples · code you cannot explain.

## 7. Honest caveats
Tested on synthetic images only (no Kaggle/ImageNet access in my environment): ResNet-18 weights download on your first run. Dataset structure unverified. A single GAN is trained (seeds vary the classifier only) — a limitation to state. Compute times are estimates. Declare AI assistance per the brief (§6).
