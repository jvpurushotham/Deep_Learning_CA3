# 10-minute presentation plan (+ Q&A)
Required: problem, key design decisions, results, and **one thing that did NOT work and how you diagnosed it**.

| # | Time | Slide | Say |
|---|---|---|---|
| 1 | 0:00–0:45 | Title + question | "Does GAN-generated data really improve recall on the rare defect class?" |
| 2 | 0:45–2:00 | Problem | Casting image → ok/defect; missed defects costly; why recall is the headline metric; hypothesis. |
| 3 | 2:00–3:15 | Data & leakage | Counts, controlled minority (train only), near-duplicate-safe split, GAN sees train minority only, test once. |
| 4 | 3:15–4:30 | Stage 1 | Scratch CNN vs ResNet-18 (transfer learning, two phases); why ResNet-18 for small data; curves. |
| 5 | 4:30–5:45 | Stage 2 | GAN as police-and-thief; DCGAN; uncurated samples; diversity / failure checks. |
| 6 | 5:45–7:15 | Results | Bar chart of test defect recall (baseline, class weights, oversampling, GAN) ± sd; bootstrap CI; answer in one sentence. |
| 7 | 7:15–8:30 | Grad-CAM | TP, FN, FP panels; border share; trust? |
| 8 | 8:30–10:00 | What did NOT work + limits | Your real failure story; 2 limitations; when not to deploy. |

Tips: use figures from `outputs/figures/`; ≤ 25 words per slide; keep backup slides (architecture, GAN objective, Grad-CAM formula, confusion matrices).

## Pick a *real* "did not work" from your run (do not invent)
* GAN samples blurry / collapsed → diversity ratio, fewer epochs, 128 px, D/G balance.
* GAN arm no better than oversampling → synthetic images add no information beyond the few real ones.
* Classifier exploited synthetic resolution (shortcut) → `--degrade-res` control.
* Baseline accuracy high but defect recall low → imbalance; class weights/threshold fix.
* Scratch CNN overfits → pre-training, augmentation, dropout.
* Grad-CAM highlighted the rim rather than the defect → spurious cue.
