# Explainable Caries Detection on Panoramic Radiographs

Academic project (thesis). Train YOLOv8 to detect caries on DENTEX, test generalisation
on the Tufts Dental Database, produce Grad-CAM/EigenCAM heatmaps and the statistics,
tables and figures for Chapter Four.

## Environment (MacBook Air M1, 16 GB, fanless)
- Conda env `caries` (Python 3.11) via Homebrew Miniforge: `conda activate caries`.
  Absolute python: `/opt/homebrew/Caskroom/miniforge/base/envs/caries/bin/python`
- torch 2.14.1 (MPS works), ultralytics 8.4.173, grad-cam, opencv, pandas, scikit-learn.
- Ultralytics `runs_dir` is set to `/Users/Felixchima/caries/runs`.
- The Mac sleeps during long jobs and training stalls. Prefix long runs with
  `caffeinate -is`, keep it on AC power with the lid open.
- Company-managed laptop on a flaky proxy: large downloads drop. Use `aria2c -c`
  (installed via brew), not plain curl or the huggingface_hub downloader.

## Data
- Raw DENTEX zips + extracted files: `data/dentex/` (from HF `ibrahimhamamci/DENTEX`).
- YOLO dataset: `data/dentex_yolo/` built by `scripts/prepare_dentex.py`, config `caries.yaml`.
  528 train / 113 val / 114 test images, single class `caries`.
- Tufts: must be downloaded manually by the user from the emailed link (terms forbid
  scripted download) into `data/tufts/`. Not downloaded yet; caries-label mapping to
  DENTEX must be checked before using it as the external set (fallback: figshare
  children's caries dataset).

## Methodology decisions (flag to supervisor, note in thesis)
- "Caries" and "Deep Caries" (DENTEX categories_3 ids 1 and 3) are merged into one
  class. `--caries-only` flag excludes Deep Caries.
- Split is per image (70/15/15, seed 42): DENTEX is anonymised, no patient IDs.
- DENTEX official test set (250 images, `data/dentex/extracted/disease/label/*.json`)
  is excluded: LabelMe polygons with a different Turkish label scheme (`çürük`=caries,
  `kanal`=root canal, no deep-caries label). Pending supervisor decision.

- Thesis Chapter 3 (user's PDF) names **Faster R-CNN + ResNet backbone as the primary
  model** and YOLO as the comparison baseline (Sec 3.6.2, 3.10). The YOLOv8s run is
  the baseline. Also promised there: early stopping, multiple seeds, Grad-CAM IoU
  fidelity (needs masks; DENTEX has boxes only), specificity (needs an image-level
  TN definition), patient-level split (we can only do per image).
- Supervisor APPROVED Ch 1-3 (6 Oct 2026) and asked for Ch 4-5. Do not edit Ch 3; implement
  what it says, and report unavoidable deviations (per-image split, Grad-CAM IoU vs boxes
  not masks) in Ch 4.

## Progress
- [x] Environment, DENTEX download, conversion, caries.yaml
- [x] Trial: yolov8n, 10 epochs, MPS -> `runs/detect/trial/` (val mAP50 0.485, P 0.42, R 0.58)
- [x] Full training: yolov8s, 100 epochs, MPS -> `runs/detect/full/` (~3.5 h). Best val mAP50 0.585,
      mAP50-95 0.39, P 0.51, R 0.58 at epoch 29; overfits after ~35 (val cls loss rises,
      epoch-100 mAP50 0.375). Use `best.pt`. Optional rerun: patience=20 / more augmentation.
- [x] YOLO baseline test eval -> `runs/detect/yolo_test/`: mAP50 0.527, mAP50-95 0.349, P 0.457, R 0.553
      (Ultralytics numbers; rescore with `scripts/metrics.py` so both models use identical code)
- [ ] Primary model: `scripts/train_frcnn.py` (Faster R-CNN R50-FPN v2, Adam, early stopping,
      `--seed`, `--resume`) -> `runs/frcnn/<name>/`. Smoke-tested; ~20 min/epoch on M1.
      Plan: seeds 0,1,2. Then predict + score test split with `scripts/metrics.py`.
      Must use `--batch 1` (batch 2 hit 11 GB RSS and swap-stalled on 16 GB). ~30-40 min/epoch.
      - [x] seed 0 -> `runs/frcnn/frcnn_s0/`: best val mAP50 0.576 at epoch 10, early-stopped
            after epoch 18 (~10 h). Overfits after ~10 (train loss falls, val drifts to 0.53).
      - [ ] seeds 1, 2: GPU queue `scripts/run_queue.sh` (log `runs/queue.log`), started 11 Oct ~10:00:
            YOLO s1 -> YOLO s2 -> FRCNN s1 -> FRCNN s2 -> evaluate each. Resumable: just rerun it.
            Ultralytics nests output in `runs/detect/runs/detect/yolo_sN`; `runs/detect/yolo_sN` are symlinks.
            11 Oct: FRCNN seeds 1, 2 moved to Google Colab T4 (`notebooks/colab_frcnn.ipynb`, results to
            Drive `caries_runs/`, user downloads `frcnn_results.zip` to ~/Downloads). Stub logs
            `runs/frcnn/frcnn_s{1,2}.log` ("done: ... Colab") make the Mac queue skip them. On import, unzip
            into runs/ (overwrites stubs) and rerun analysis.py. Note in Ch 4: seeds 1-2 trained on CUDA T4.
- [x] Test scoring: `scripts/evaluate.py {frcnn|yolo} <weights> --name X` -> `runs/eval/X_test.json`,
      raw preds cached in `runs/eval/X_{val,test}.npz`. Threshold = max-F1 on val, applied to test.
      YOLO mAP50 0.527 matches Ultralytics (validates metrics.py). Seed-0 test: frcnn mAP50 0.551, F1 0.532;
      yolov8s mAP50 0.527, F1 0.485. Only 11 caries-free test images: specificity unreliable (Wilson CI).
- [x] `scripts/analysis.py` -> `runs/analysis/`: per-seed + mean±SD, paired bootstrap (2000) FRCNN-YOLO,
      FROC/CPM, D-ECE calibration, image-level Wilson CIs, internal-vs-external drop CIs (IoU 0.5 and 0.3),
      all figures. Seed 0: mAP50 diff +0.025 p=0.13 (n.s.); F1 +0.047 p=0.007; precision p=0.001;
      mAP50-95 p=0.02; CPM p=0.04. FRCNN overconfident (D-ECE 0.19), YOLO underconfident (0.13).
      RERUN after the queue finishes (picks up all seeds automatically).
- External set: Tufts unusable for box-level caries (one generic "abnormality" mask per image,
      caries only in free text). Chosen: (1) children's set, Zhang et al. 2023 Sci Data (figshare, CC BY),
      downloaded + converted (`scripts/prepare_external.py children` -> `data/external/children_yolo`,
      100 imgs, 314 caries boxes, 27 caries-free; only label 龋病 kept). Population shift (ages 2-13),
      so a stress test. (2) PREFERRED adult set: Muresanu et al. 2025 Romania, Zenodo 15487430
      (180-image multi-centre external split, YOLO, class 4 = CAR, 296 caries). NOT downloaded:
      custom terms (no sharing, non-commercial) the user must accept. 1.4 GB, md5 18d5acaea021f4f3b05b76e9a27fbd42.
      Children result seed 0 (DENTEX-val thresholds): FRCNN mAP50 0.018 (IoU .3: 0.204), YOLO 0.092
      (IoU .3: 0.300). Not a bug: models predict whole-tooth boxes (~130x225 px) as on DENTEX, children GT
      boxes are smaller (~84x124, crown/lesion), so box convention + age shift drive the drop. YOLO
      generalises better than FRCNN here (reverses the internal ranking). Image-level sens 0.86/0.95.
- [ ] Explainability faithfulness study `scripts/explain.py` -> `runs/explain/` (CPU, ~10 h, started
      11 Oct ~10:20, log `runs/explain.log`, resumable). FRCNN: Grad-CAM, HiResCAM, EigenCAM, RISE
      (box-score target), random + random-in-box controls; YOLO EigenCAM. Localisation, deletion/insertion
      AUC, Adebayo randomisation sanity (whole image + in box). Smoke-test findings: standard Grad-CAM
      fails for the detector (global-average weights wash out the RoI signal; peaks at image border);
      HiResCAM localises but box-target gradient maps sit in the box by construction, so judge them with
      the in-box controls; EigenCAM unchanged when heads are randomised (does not explain the caries decision).
- [x] Reader study kit `scripts/reader_study.py make|score` -> `reader_study/` (50 imgs: 11 negatives +
      39 random positives; send only images/ + INSTRUCTIONS.md; makesense.ai YOLO export into
      readers/<initials>/). Needs 2+ clinicians recruited by the user.
- [x] Repo: README.md, requirements.txt, .gitignore (data, weights, reader key excluded), `git init`.
      Not committed or pushed: user to confirm making it public on GitHub.
- [ ] Chapter 4 brief/draft refresh with all results (the earlier chapter4_brief.md was moved/deleted).
- [ ] PhD extension: paper on explanation faithfulness for dental detectors (targets: DMFR, J Dent,
      BMC Oral Health, Sci Rep, MIDL short paper); preprint.
