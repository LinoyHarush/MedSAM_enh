# AlphaDent: external test (E3), work plan

Written 2026-10-20. External test of our final models on a second, independent dental dataset: another clinic, another camera (DSLR) and other annotators. **Inference only: nothing is trained, tuned or chosen on AlphaDent.** Every setting is already fixed from our own data: zoom above 2,000 px, crop margin 0.5, the boxes rule and the model checkpoints. That is what makes it a true external test.

Code: `MedSAM/prepare_alphadent.py` (conversion), `MedSAM/colab_extras.py` (`check_alphadent`, `alphadent_plan`, `score_set`, `r7_vs_r1`), `MedSAM/colab_runs.py` (`score`, `save`, `test_table`, `final_table`), `MedSAM/colab_figures.py` (figures with a dataset argument). The same overwrite guards apply: every result gets a new name under `alphadent_valid`.

## The dataset

AlphaDent (Sosnin et al. 2025, [arXiv 2507.22512](https://arxiv.org/pdf/2507.22512)):
- **Images:** 1,320 DSLR intraoral photos (Canon 6D Mark II, macro lens, intraoral mirror) from 295 patients. Most are larger than 5,000 × 3,000 px, so zoom applies to every photo.
- **Labels:** drawn by dentists. Per image there is a YOLO segmentation `.txt` (one row per object: `class x1 y1 x2 y2 …`, normalised) and a folder of PNG masks named like `00_class_3.png`.
- **Classes:** 0 abrasion, 1 filling, 2 crown, and **3–8 caries** (Caries 1–6, by tooth surface).
- **Splits:**
  - `train` (273 patients) and `valid` (22 patients, about 83 photos): labels available.
  - `test`: labels closed (Kaggle leaderboard only), so we can't use it.

**Download:** Zenodo record 16582489, https://zenodo.org/records/16582489 (from our project notes; open the page once in a browser to confirm the files and the licence). Cell AD1 lists the record's files through Zenodo's API before downloading anything. The training code and weights are at https://github.com/ZFTurbo/AlphaDent and the competition at https://www.kaggle.com/competitions/alpha-dent/.

**Where it goes:**
- **On Drive:** only the downloaded archive(s), in `MedSAM_enh/data/raw/alphadent/` (outside the processed data, never edited).
- **On the runtime:** the unzipped files and the converted npz files live only on the runtime's disk (`/content/alphadent/`, `/content/work/data/alphadent_valid/`). They're several GB, and re-creating them takes minutes.
- **Results:** scores, predictions, tables and figures go to Drive as usual, under `experiments/alphadent_valid/` and `data/alphadent_valid/…/segs_*`.

## Which photos, and what counts as the result (fixed now, before any AlphaDent result is seen)

- **Main set: AlphaDent `valid`, all photos that contain caries.** It's split by patient, as the authors intended, and has about 83 photos.
- **Optional, only if time remains: a fixed random sample of 150 `train` photos** (seed 0, `-n 150 -seed 0`), as a larger second set with the same cells (`alphadent_train`). The decision to run it doesn't depend on the `valid` results.
- **Main comparison:** R7 (per photo, mean of 3 seeds) vs Z, boxes as given, two-sided Wilcoxon.
- **Also reported:** every model vs Z and vs B0 (Holm-corrected), the 20% looser boxes, R7 vs R1, and the same figures as on our test set.
- **Not comparable with the AlphaDent paper's numbers.** They report detection mAP for a fully automatic model (merged caries: mAP50 0.316 on `valid`), while we give the model the boxes and measure DSC. We say so in the report.

## How the binary mask matches ours

| | Our dataset (Roboflow) | AlphaDent (as converted) |
|---|---|---|
| Foreground (1) | caries, cavity, crack (classes 1–3), merged | caries 1–6 (classes 3–8), merged |
| Background (0) | tooth and everything else | abrasion, filling, crown and everything else |
| Boxes | one per annotated lesion | one per caries polygon (tight box around the rasterised polygon) |
| Tiny objects | lesions of 5 px or less have no box | caries polygons under 6 px are dropped from boxes and mask |
| Overlaps | union in the mask | union in the mask |
| Photos without any lesion | not prompted | skipped (listed in `photos.csv`) |

**One real difference to report:** AlphaDent has no "crack" class, and its caries classes are defined per tooth surface. The annotators may outline caries differently from Roboflow's. A score drop on AlphaDent can therefore partly reflect labelling style, not only the model. Step AD5 (b) makes sure our conversion itself is exact.

## Steps

| Step | What | CPU / GPU | Time (estimate) |
|---|---|---|---|
| AD0 | Session setup | CPU | 5 min |
| AD1 | List the Zenodo files | CPU | 1 min |
| AD2 | Download to the runtime, keep a copy on Drive, unzip | CPU | 15–45 min (several GB) |
| AD3 | Check the class list | CPU | 1 min |
| AD4 | Convert `valid` to our layout | CPU | 5–10 min |
| AD5 | Check the conversion | CPU | 5 min |
| AD6 | Inference and scoring for all models | **GPU recommended** (CPU possible but several hours) | 1.5–2.5 h on a T4, mostly computing NSD on 15-MP masks |
| AD7 | Tables | CPU | 2 min |
| AD8 | Figures | CPU | 5 min |

AD0–AD5 can run in a CPU session. AD6–AD8 need the converted photos on the runtime's disk. In a new GPU session, repeat AD0, AD2 (copy from Drive, no download) and AD4, which takes minutes.

### AD0 · Session setup (CPU)
Daily setup cell, pip cell, S0 (GPU runtime only for AD6), S1, and the first three lines of S2:
```python
import sys; sys.path.insert(0, "/content/code/MedSAM")
import colab_runs as cr
cr.init(REPO_D)
import importlib, colab_extras as cx, colab_figures as cf
importlib.reload(cx); importlib.reload(cf)
AD = "alphadent_valid"
```

### AD1 · List the files in the Zenodo record (CPU)
```python
import requests
rec = requests.get("https://zenodo.org/api/records/16582489", timeout=60).json()
print(rec["metadata"]["title"], "| licence:", rec["metadata"].get("license", {}).get("id"))
for f in rec["files"]:
    print(f"{f['key']:40s} {f['size'] / 1e9:6.2f} GB")
```
Check that the title is AlphaDent and note the licence. If the record holds several archives, download only the one(s) with `images/` and `labels/`. If unsure, send Claude the list.

### AD2 · Download (once) and unzip (CPU)
```python
import os
RAW_D = f"{REPO_D}/data/raw/alphadent"
os.makedirs(RAW_D, exist_ok=True); os.makedirs("/content/alphadent_raw", exist_ok=True)
FILES = [f for f in rec["files"] if f["key"].endswith(".zip")]       # adjust if AD1 shows other names
for f in FILES:
    local = f"/content/alphadent_raw/{f['key']}"
    if os.path.exists(f"{RAW_D}/{f['key']}"):                         # already on Drive: copy, don't download
        cr.sync(f"{RAW_D}/{f['key']}", "/content/alphadent_raw/")
    else:
        cr.sh(f'wget -q --show-progress -O "{local}" "{f["links"]["self"]}"')
        cr.sync(local, f"{RAW_D}/")                                     # keep the archive on Drive
    cr.sh(f'unzip -q -o "{local}" -d /content/alphadent')
cr.sh("find /content/alphadent -maxdepth 3 -type d | head -30")
```
Expect folders `images/train`, `images/valid`, `labels/train` and `labels/valid` somewhere under `/content/alphadent`. If the download from Zenodo fails, download the archive in your browser, put it in `MedSAM_enh/data/raw/alphadent/` on Drive, and run the cell again: it copies instead of downloading.

### AD3 · Check the class list (CPU)
```python
import glob, yaml
y = yaml.safe_load(open(glob.glob("/content/alphadent/**/*.yaml", recursive=True)[0]))
names = y["names"]; names = dict(enumerate(names)) if isinstance(names, list) else names
for k, v in sorted(names.items(), key=lambda kv: int(kv[0])):
    print(k, v, "<- caries (foreground)" if int(k) in (3, 4, 5, 6, 7, 8) else "")
```
Classes 3–8 must be the six caries classes, and 0–2 abrasion, filling and crown. **If not, stop and tell Claude**: the `-caries` argument in AD4 must match.

### AD4 · Convert `valid` to our layout (CPU)
```python
cr.sh("cd /content/code/MedSAM && python prepare_alphadent.py -src /content/alphadent -split valid -out_root /content/work/data")
```
It prints:
- photos read and written;
- photos skipped because they have no caries;
- the number of caries boxes (the paper reports 258 caries instances on `valid`, so expect a number near that);
- tiny instances dropped (expect very few);
- how many photos are above 2,000 px (expect all of them).

For the optional larger set: `-split train -n 150 -seed 0`, with `AD = "alphadent_train"` in the later cells.

### AD5 · Check the conversion before running any model (CPU)
```python
ious = cx.check_alphadent("/content/alphadent", ds=AD, split="valid")
```
| Check | Pass if |
|---|---|
| (a) counts | photos and caries boxes as in AD4 |
| (b) our binary caries mask vs AlphaDent's own caries PNG masks, 10 random photos | IoU median ≥ 0.98 and no "ROTATION/SIZE MISMATCH" line. A mismatch means the photos and labels disagree on orientation: stop and tell Claude. **The Zenodo archive has no PNG masks** (only `images/` and `labels/`), so (b) is replaced by (b1) and (b2) |
| (b1) EXIF orientation of every kept photo | all unrotated (tag missing or 1). YOLO labels refer to the photo after EXIF rotation, while we read the pixels as stored; they agree only if no photo is rotated. Any "ROTATED" line: stop and tell Claude |
| (b2) label coordinates | 0 outside [0, 1] |
| (c) boxes with no caries pixels | 0 |
| (d) the overlay figure (`experiments/alphadent_valid/figures/check_conversion.png`) | green masks sit on carious spots, and every dashed box surrounds one |

Only go on to AD6 if all of them pass.

**Status (2026-10-21, CPU session):**
- AD1: one archive, `AlphaDent.zip`, 4.88 GB, licence CC-BY-4.0. It's now also on Drive in `data/raw/alphadent/`.
- AD3: classes as expected (0 Abrasion, 1 Filling, 2 Crown, 3–8 Caries 1–6).
- AD4: 83 photos read, 64 with caries written, 19 without caries skipped. **258 caries boxes, exactly the paper's 258 caries instances on `valid`**; 0 tiny instances dropped; long side median 5,472 px, all 64 above 2,000 px.
- AD5: (a) passed. (b) found no PNG masks. (c) and (d) didn't print in that run. Re-run AD5 after pulling the new `colab_extras.py` (b1/b2 added) and check (b1), (b2), (c) and (d).
- **AD5 re-run 2026-10-22: all passed.** (b1) 64 of 64 photos unrotated; (b2) 0 of 763,526 label coordinates outside [0, 1]; (c) 0 boxes without caries; (d) the overlay shows the masks on tooth surfaces inside their boxes. **Green light for AD6.**
- Noted from the overlay, for the report: AlphaDent's caries outlines are coarse polygons (often 3–5 corners), much less detailed than Roboflow's. A model that follows the real lesion edge can't match them closely, which caps DSC and NSD on AlphaDent for every method. That's one reason the numbers may be lower than on our test set, apart from the new clinic and camera.

### AD6 · Inference and scoring (GPU recommended)
Runs:
- **B0:** whole photo, saved as `og_ckpt` and `B0_p20`;
- **Z;**
- **R1 and R7:** 3 seeds each;
- **R2, R3, R6:** seed 0.

Each is run with the boxes as given and 20% looser, the same names as on our test set. Models already scored are skipped, so re-run the cell after a reset (after AD0, AD2 and AD4).
```python
cx.score_set(AD, cx.alphadent_plan())
```
Watch the first model: if one model takes more than about 15 minutes in total, the NSD computation on the 15-MP masks is the slow part. Tell Claude rather than stopping it.

### AD7 · Tables (CPU)
```python
ALL = ["R1_s0", "R1_s1", "R1_s2", "R7_s0", "R7_s1", "R7_s2", "R2_s0", "R3_s0", "R6_s0"]
t_ad = cr.test_table(ALL, ds=AD, out_name="alphadent_comparison.csv")     # every run vs Z and vs B0, Holm
f_ad = cr.final_table({"R1": ALL[:3], "R7": ALL[3:6]}, ds=AD)            # means ± sd over seeds
cx.r7_vs_r1(AD)                                                          # R7 vs R1, per-photo seed means
```
The main line is R7 vs Z with boxes as given (in `t_ad`, use the per-seed rows, plus the seed means from `f_ad`).

### AD8 · Figures (CPU)
```python
cf.fig_overview(AD)       # every method on AlphaDent, boxes as given
cf.fig_robustness(AD)     # boxes as given vs 20% looser
cf.fig_examples(AD)       # 4 photos chosen by rule (R7's biggest gains over Z, zoom's biggest gain over B0, R7's worst loss)
```
Saved in `experiments/alphadent_valid/figures/`. Send Claude the AD4 printout, the AD5 checks, the AD7 tables and the figures.

## Total time

About **half a day of work spread over one or two sessions**:
- **CPU:** ~1 h (AD0–AD5, AD7, AD8);
- **GPU:** ~2 h, mostly waiting (AD6);
- **Optional `train` sample:** about another 2–3 h of GPU.

**Main risks:**
- the archive's exact file names (AD1 shows them before anything is downloaded);
- orientation of the photos vs the labels (AD5 checks it);
- Drive dropping during the multi-GB copy (`cr.sync` retries);
- slow NSD on very large masks (AD6).
