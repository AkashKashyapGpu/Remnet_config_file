# REMNET — Waste Image Classification, External Validation, and Sustainability

Research code and archived outputs for **REMNET** (ensemble of MobileNetV2, ResNet50, and EfficientNet-Lite0), including original-dataset classification, **separately run** CodeCarbon sustainability experiments, and frozen-model zero-shot evaluation on two additional image collections. The repository contains 29 Jupyter notebooks, one standalone dataset-audit Python module, generated CSVs and CodeCarbon logs. The updated package includes a **directory scaffold** matching the newly photographed checkpoint layout, but does **not** include original image datasets or the trained `.pt` checkpoint bytes. The screenshot-only additions must be copied from the user's local computer before checkpoint-based reproduction.

> **Critical experimental distinction.** The original **classification** and **sustainability** workflows are **separate training/inference experiments**, not two measurements of exactly the same training execution. The recorded classification accuracies must not be combined with independently monitored energy records as though the model weights, execution duration, and checkpoint identities were paired. In particular, REMNET's classification and sustainability constituent checkpoint sets differ; the standalone ResNet50 implementations also differ (`torchvision` versus `timm`). Energy/accuracy or carbon/accuracy quotients assembled from their separate means are derived cross-run indicators, **not validated checkpoint-matched efficiency estimates**. Some D3 sustainability diagnostic accuracies also differ from classification results. Do not overwrite or silently reconcile those values.

## Directory map (updated to the 20 November 2026 local layout)

```text
final_code/
├── README.md
├── requirements.txt
├── config.yaml
├── LICENSE
├── Checkpoints_trained_model/
│   ├── README.md                  # checkpoint inventory and provenance warnings
│   └── checkpoints/
│       ├── Classification/
│       │   ├── efficientnet_lite0/
│       │   ├── mobilenet_v2/
│       │   ├── Remnet/            # 3 × 5 constituent checkpoint files in local screenshot
│       │   └── Resnet_50/
│       └── Sustainability/        # separate checkpoint experiment
│           ├── efficientnet_lite0/
│           ├── mobilenet_v2/
│           ├── Remnet/
│           └── Resnet_50/
├── REMNET/
│   ├── 01_Classification_Implementation/
│   ├── 02_Energy_and_Carbon_Measurement/
│   ├── 03_Statistical_Analysis/
│   ├── 04_Classification_Results/
│   ├── 05_Computational_Complexity/
│   └── 06_Dataset_Preparation/     # archived audit + place for new Stage 1 manifests
├── 09_Sustainability_Results/      # original CodeCarbon logs and source CSV
├── Garbage Classification V2 external validation/
│   └── 07_Garbage_Classification_V2/
│       ├── Dataset verification/
│       ├── Classification implementation/
│       └── Classification results/  # REMNET CSVs + locally shown All models.zip
└── RealWaste external validation/
    └── 08_RealWaste/
        ├── Dataset verification/
        ├── Classification implementation/
        ├── Classification results/  # REMNET D3 CSVs + locally shown all models.zip
        └── Sustainability implementation/
```

> **What is included versus shown locally:** In the downloadable updated repository, all existing notebooks, results, historical split-audit files and CodeCarbon logs remain intact. The `Checkpoints_trained_model` directory structure is provided, but original `.pt` files **are not** bundled. The new Stage 1 split-evidence files and the two `All models.zip`/`all models.zip` nested archives are visible in your screenshots **only**; they were not supplied as bytes in the previous repository ZIP. See [`Checkpoints_trained_model/README.md`](Checkpoints_trained_model/README.md), [`README_Stage1_Local_Update.md`](REMNET/06_Dataset_Preparation/README_Stage1_Local_Update.md), and the classification-results folder notices before publishing.

### New locally photographed files and where to place them

| Source shown in screenshots | Intended location within repository | Status in this updated downloadable ZIP |
|---|---|---|
| Classification checkpoint folders (`efficientnet_lite0`, `mobilenet_v2`, `Remnet`, `Resnet_50`) | `Checkpoints_trained_model/checkpoints/Classification/` | Folder scaffold; binary weights not supplied |
| Sustainability checkpoint folder | `Checkpoints_trained_model/checkpoints/Sustainability/` | Folder scaffold; binary weights not supplied |
| 15 REMNET constituent `.pt` files (three CNNs × five seeds) | `Checkpoints_trained_model/checkpoints/Classification/Remnet/` | **Not supplied**; filenames documented |
| New retained/split evidence: `REMNET_All_2906_Locked_Dataset_Splits...`, `REMNET_Final_Retained_Split_Manifest.csv`, `REMNET_Training_Split.csv`, `REMNET_Validation_Split.csv`, `REMNET_Testing_Split.csv`, classwise counts and verification JSON | `REMNET/06_Dataset_Preparation/` | **Not supplied**; older audit files retained |
| `All models.zip` (V2) and the existing two REMNET comparison CSVs | `Garbage Classification V2 external validation/07_Garbage_Classification_V2/Classification results/` | Existing REMNET CSVs retained; model ZIP not supplied |
| `all models.zip` (RealWaste) and the existing two REMNET D3 CSVs | `RealWaste external validation/08_RealWaste/Classification results/` | Existing REMNET CSVs retained; model ZIP not supplied |

**Important:** The labels `Classification` and `Sustainability` identify **separate experiments and checkpoint lineages**, not two validated measurements from one checkpoint. The nested `Remnet` directory shows component `.pt` files rather than a single ensemble-weight file. Confirm their SHA-256 hashes, checkpoint architectures and per-seed identities before making any matched-run claim.

The historical D1 classification evaluation uses **463 images**, while an earlier independently filtered dataset audit recorded **454 images**. The new Stage 1 split screenshots show extra manifests, but their data have not been inspected here. Until the exact file membership is checked, do not infer that those different split counts refer to identical cohorts.

## Table 15 — architecture-level inference profiling (untrained models)

The notebook [`REMNET/05_Computational_Complexity/throughput.ipynb`](<REMNET/05_Computational_Complexity/throughput.ipynb>), used for **Table 15 in the manuscript**, is an **architecture-level, inference-only GPU benchmark**, **not an evaluation of trained REMNET/classifier checkpoints**. The code constructs MobileNetV2, EfficientNet-Lite0, and ResNet50 with **randomly initialized weights** (`weights=None` or `pretrained=False`) and runs their three-model REMNET soft-voting ensemble. **No classification checkpoints are loaded, and no training or fine-tuning is performed.**

- **Measured:** synchronized forward-pass **batch-size-1 latency** (mean and 95th percentile), **batch-size-32 throughput**, and **peak allocated GPU memory** on the original dataset's audited image cohort, resized to **32 × 32 RGB**. The notebook uses **five profiling repetitions**, **not five independently trained seeds**. Batch-1 timing uses the first 128 audited test images per repeat by default; batch-32 throughput uses the complete 463-image audited test cohort per repeat.
- **Not measured or claimed:** trained-model classification accuracy, the inference latency of the exact trained D1/D3 checkpoints, end-to-end decoding/data-transfer latency, actual deployment/edge-device performance, or any improvement attributable to the learned weights. These profile numbers **must not be combined with classification accuracy or separate CodeCarbon training-energy results as if obtained from an identical trained execution**.
- **Real-time label:** "capable" means *inference-only* batch-1 p95 forward latency ≤ **33.33 ms**, the nominal 30-FPS frame-time budget, **on the tested GPU and at the tested resolution only**. It does not certify an end-to-end 30-FPS application.
- **Numbering note:** the executable notebook and its exported files presently label this measurement **`Table14_*`** (reflecting an earlier table number). The manuscript refers to the same architecture-only profiling results as **Table 15**. Rename outputs only if you also update manuscript references consistently; **do not interpret the labels as different experiments**.

For a deployment-oriented comparison of **trained** models, run a separate, clearly labeled benchmark that loads the exact frozen classification checkpoints and documents checkpoint hashes, preprocessing, GPU environment and input workload. Report that benchmark separately from these architecture-only measurements and from independently monitored sustainability experiments.

## Datasets and download links

| Role | Source and download | Dataset path used in the notebooks | Restrictions and caveats |
|---|---|---|---|
| D1 — original development/test | [Garbage Classification Image Dataset (Kaggle)](https://www.kaggle.com/datasets/saumyamohandas/garbage-classification-image-dataset) | `/kaggle/input/datasets/saumyamohandas/garbage-classification-image-dataset` (some notebooks also accept `/kaggle/input/garbage-classification-image-dataset`) | Six material classes; attach the dataset in Kaggle. Notebook-saved original-test results refer to **463 images**, while the included standalone `REMNET_split_summary.json` reports **454 final test images after its own filtering**. These are different recorded split artifacts, and must not be assumed equivalent. |
| V2 — external screened cohort | [Garbage Classification V2 (Kaggle)](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2), **version 12** | `/kaggle/input/datasets/sumn2u/garbage-classification-v2` | 10 source classes; six mapped classes; overlap/pHash screening yields **4,931** retained images. Common-source provenance with D1 may remain despite screening. The archived audit records an unresolved license discrepancy: Kaggle notice **MIT** vs creator site **CC BY-SA 4.0**. Check source terms before distributing images. |
| D3 — RealWaste external cohort | [Official RealWaste dataset (UCI)](https://archive.ics.uci.edu/dataset/908/realwaste) ([DOI](https://doi.org/10.24432/C5SS4G)); [mounted Kaggle mirror](https://www.kaggle.com/datasets/akash264208/real-waste) | `/kaggle/input/datasets/akash264208/real-waste` | UCI source license recorded as **CC BY 4.0**; mirror revision not verified. Map *Miscellaneous Trash* to *trash*. The locked six-class cohort contains **3,497** images after conservative screening. |

**Class order:** `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`. For D3, confirm the recorded mapping CSV before inference. Attach the original image datasets and use the **existing audited manifest**, not a new convenience split or an arbitrary sample of images.

Dataset provenance files are included alongside the verification notebooks:
- `Garbage Classification V2 external validation/07_Garbage_Classification_V2/Dataset verification/Section2_Source_License_Citation_Version_Download_Details.json`
- `RealWaste external validation/08_RealWaste/Dataset verification/D3_Section2_Source_License_Citation_Version_Download_Details.json`

### Populate local additions before re-running checkpoint-based evaluations

After extracting this archive, copy your local files from `final_code_Remnet/final_code/` into the corresponding paths above:

1. Copy the **classification** `.pt` files into `Checkpoints_trained_model/checkpoints/Classification/` under their actual model directory; copy **sustainability** `.pt` files only into `.../Sustainability/`. Keep original filenames and record SHA-256 hashes.
2. Copy the new Stage 1 evidence CSV/JSON/README files into `REMNET/06_Dataset_Preparation/` without deleting or silently rewriting historical audit records.
3. Copy the genuine local `All models.zip` (V2) and `all models.zip` (RealWaste) into their respective `Classification results/` directories if those result packages are intended for distribution.
4. Confirm each notebook's hard-coded Kaggle input path resolves to the intended mounted dataset and checkpoint collection. Neither `config.yaml` nor this directory scaffold changes notebook code automatically.

No checkpoint or split-file content was inferred from the screenshots or fabricated in this archive. The checkpoint README names the files visible in the local `Remnet` screenshot and describes the evidence required for safe reuse.

### Kaggle preparation

1. Create a Kaggle notebook with a GPU (Tesla T4 was recorded for inference measurements) when running deep-learning notebooks; CPU is sufficient for most audits and summary statistics.
2. Add the original Kaggle dataset to **Input**. For V2 also add `sumn2u/garbage-classification-v2` revision 12 or allow `kagglehub` to fetch it. For D3 attach the RealWaste Kaggle mirror, or stage the officially downloaded UCI images in the exact class-root layout expected by the code.
3. Supply the **separate, private** checkpoint/manifest dataset needed by zero-shot notebooks; the source archive does not contain these `.pt` weights. V2/D3 **classification** notebooks expect `/kaggle/input/datasets/akash264208/checkpoint-manifist`, with `Classification_checkpoints`, `manifist`, and `per_seed_performance_summary` discoverable beneath it. D3 **sustainability** notebooks instead use `/kaggle/input/datasets/akash264208/manifist-part2/Section3` and its `manifist` subdirectory. Confirm these inputs are present before running. The two roots are **not automatically interchangeable**.
4. Install dependencies (Kaggle notebooks may already include them): `python -m pip install -r requirements.txt`. For PyTorch CUDA wheels, use a platform-specific compatible build, ideally the recorded Kaggle environment; see version notes below. Kaggle installs may be notebook-local.
5. Run each notebook **top to bottom** in its intended Kaggle environment. Files and figures are written under `/kaggle/working/`; download the generated reports/CSVs/ZIPs at the end of a run. The notebooks embed their own paths and constants: **editing `config.yaml` alone will not change them**.

## How to run each notebook

Run the following notebooks separately. These paths are relative to `final_code/` and are case-sensitive on Linux/Kaggle.

### A. Original-dataset (D1) workflow

**Source-data audit (optional, no training):**

```bash
python "REMNET/06_Dataset_Preparation/remnet_dataset_audit.py" \
  /kaggle/input/datasets/saumyamohandas/garbage-classification-image-dataset \
  /kaggle/working/REMNET_dataset_audit
```

The script's filtering/split decisions can differ from the historical 463-image evaluation split. Preserve original manifests and reconcile the cohort counts before using the audit outputs to reproduce published figures. The historical evaluation notebooks may rely on an external original split manifest that is not included in this ZIP.

| Notebook (open in Kaggle and **Run all**) | Purpose |
|---|---|
| [`REMNET/01_Classification_Implementation/Efficientnet_lite0_accu.ipynb`](<REMNET/01_Classification_Implementation/Efficientnet_lite0_accu.ipynb>) | Train/evaluate standalone EfficientNet-Lite0 on D1. |
| [`REMNET/01_Classification_Implementation/MobileNet_Accu_s2.ipynb`](<REMNET/01_Classification_Implementation/MobileNet_Accu_s2.ipynb>) | Train/evaluate standalone MobileNetV2 on D1. |
| [`REMNET/01_Classification_Implementation/Resnet_50_Acc_s2.ipynb`](<REMNET/01_Classification_Implementation/Resnet_50_Acc_s2.ipynb>) | Train/evaluate standalone ResNet50 on D1. |
| [`REMNET/01_Classification_Implementation/Remnet_Accu_s2.ipynb`](<REMNET/01_Classification_Implementation/Remnet_Accu_s2.ipynb>) | Train/evaluate REMNET's three CNN constituents and ensemble on D1. |
| [`REMNET/02_Energy_and_Carbon_Measurement/efficientnet-lite-prob-co2-s2.ipynb`](<REMNET/02_Energy_and_Carbon_Measurement/efficientnet-lite-prob-co2-s2.ipynb>) | **Separate** EfficientNet-Lite0 training/inference and CodeCarbon tracking. |
| [`REMNET/02_Energy_and_Carbon_Measurement/mobilenet-v2-prob-c02-s2.ipynb`](<REMNET/02_Energy_and_Carbon_Measurement/mobilenet-v2-prob-c02-s2.ipynb>) | **Separate** MobileNetV2 training/inference and CodeCarbon tracking. |
| [`REMNET/02_Energy_and_Carbon_Measurement/resnet-50-prob-co2-s2.ipynb`](<REMNET/02_Energy_and_Carbon_Measurement/resnet-50-prob-co2-s2.ipynb>) | **Separate** ResNet50 training/inference and CodeCarbon tracking. |
| [`REMNET/02_Energy_and_Carbon_Measurement/remnet-co2-prob-s2.ipynb`](<REMNET/02_Energy_and_Carbon_Measurement/remnet-co2-prob-s2.ipynb>) | **Separate** REMNET constituent training and ensemble inference tracking; the ensemble's training total should count each constituent training log once. |
| [`REMNET/03_Statistical_Analysis/REMNET_Classification_Paired_T_Test_Stage2.ipynb`](<REMNET/03_Statistical_Analysis/REMNET_Classification_Paired_T_Test_Stage2.ipynb>) | Analyze saved five-seed D1 classification measurements; inspect embedded paths and values first. |
| [`REMNET/03_Statistical_Analysis/REMNET_Sustainability_Paired_T_Test_Stage2.ipynb`](<REMNET/03_Statistical_Analysis/REMNET_Sustainability_Paired_T_Test_Stage2.ipynb>) | Analyze separately recorded energy/carbon measurements; do not treat these as weight-matched classification runs. |
| [`REMNET/05_Computational_Complexity/Gflops.ipynb`](<REMNET/05_Computational_Complexity/Gflops.ipynb>) | Profile convolution/linear FLOPs at the notebook's chosen resolution and profiling convention. |
| [`REMNET/05_Computational_Complexity/throughput.ipynb`](<REMNET/05_Computational_Complexity/throughput.ipynb>) | **Manuscript Table 15: architecture-level inference benchmark only.** Uses randomly initialized models (**no trained classification checkpoints**), measuring batch-1 latency/p95 and batch-32 throughput/peak GPU memory. Notebook output files currently use `Table14_*` names; these refer to the same profiling experiment, not a separate trained-model result. |

**Seeds** used in the five-seed evaluations: `7, 49, 112, 345, 476`. The original Stage 2 classification and carbon notebooks resize inputs to **32 × 32**. For emissions, retain *native CodeCarbon* CO₂e separately from fixed-factor `energy (kWh) × 0.475 kg CO₂e/kWh` values.

### B. Garbage Classification V2 external evaluation

1. [`Dataset verification/stage2_part1.ipynb`](<Garbage Classification V2 external validation/07_Garbage_Classification_V2/Dataset verification/stage2_part1.ipynb>) — discover/download D1 and V2 sources and inventory classes.
2. [`Dataset verification/s3p2.ipynb`](<Garbage Classification V2 external validation/07_Garbage_Classification_V2/Dataset verification/s3p2.ipynb>) — screen exact overlaps and conservative near-duplicates; produce/verify the **4,931-image** six-class manifest. Use the frozen included manifest for reproducibility.
3. With the **classification** checkpoints attached, run all four independent inference notebooks (no retraining/fine-tuning):
   - [`efficientnetlite0.ipynb`](<Garbage Classification V2 external validation/07_Garbage_Classification_V2/Classification implementation/efficientnetlite0.ipynb>)
   - [`mobilenet-v2.ipynb`](<Garbage Classification V2 external validation/07_Garbage_Classification_V2/Classification implementation/mobilenet-v2.ipynb>)
   - [`resnet-50.ipynb`](<Garbage Classification V2 external validation/07_Garbage_Classification_V2/Classification implementation/resnet-50.ipynb>)
   - [`remnet123.ipynb`](<Garbage Classification V2 external validation/07_Garbage_Classification_V2/Classification implementation/remnet123.ipynb>)
4. [`classification-test.ipynb`](<Garbage Classification V2 external validation/07_Garbage_Classification_V2/Classification implementation/classification-test.ipynb>) — run the saved statistical-analysis template after confirming its input table corresponds to the produced V2 five-seed outputs.

**Note:** V2 screening eliminates identified image overlaps but does not prove independently collected image provenance. License notices conflict in the archive; consult the current dataset author before redistribution.

### C. RealWaste D3 external evaluation

1. [`Dataset verification/d3-s2.ipynb`](<RealWaste external validation/08_RealWaste/Dataset verification/d3-s2.ipynb>) — verify source, label mapping, D1 overlap checks, and the locked **3,497-image** manifest. The audit conservatively withholds flagged pHash candidates; it does not claim human near-duplicate adjudication.
2. With **classification** checkpoint inputs attached, run the four frozen model evaluations:
   - [`efficientnetlite0-class-d3.ipynb`](<RealWaste external validation/08_RealWaste/Classification implementation/efficientnetlite0-class-d3.ipynb>)
   - [`mobilenet-class-d3.ipynb`](<RealWaste external validation/08_RealWaste/Classification implementation/mobilenet-class-d3.ipynb>)
   - [`resnet50-class-d3.ipynb`](<RealWaste external validation/08_RealWaste/Classification implementation/resnet50-class-d3.ipynb>)
   - [`remnet-class-d3.ipynb`](<RealWaste external validation/08_RealWaste/Classification implementation/remnet-class-d3.ipynb>)
3. [`class_test.ipynb`](<RealWaste external validation/08_RealWaste/Classification implementation/class_test.ipynb>) — run saved D3 classification statistical analysis; verify embedded input values against exported CSVs.
4. In **separate** Kaggle sessions with the **sustainability** checkpoint/manifest mount, run the four independently monitored D3 inference notebooks:
   - [`efficient-co2-d3.ipynb`](<RealWaste external validation/08_RealWaste/Sustainability implementation/efficient-co2-d3.ipynb>)
   - [`mobilenetv2-co2-d3.ipynb`](<RealWaste external validation/08_RealWaste/Sustainability implementation/mobilenetv2-co2-d3.ipynb>)
   - [`resnet-50-co2-d3.ipynb`](<RealWaste external validation/08_RealWaste/Sustainability implementation/resnet-50-co2-d3.ipynb>)
   - [`remnet-sustainable-d3.ipynb`](<RealWaste external validation/08_RealWaste/Sustainability implementation/remnet-sustainable-d3.ipynb>)

D3 classification and sustainability each resize to the same input size **for a given model**: EfficientNet-Lite0 and MobileNetV2 **384 × 384**, ResNet50 **350 × 350**, REMNET **484 × 484**. This is **not** the 32 × 32 Stage 2 training/monitoring size. The D3 sustainability sessions use 2 passes × 3,497 images = **6,994 processed images per seed**; they measure inference (not training) energy. Check for prediction/accuracy discrepancies between the separately executed classifications and sustainability diagnostics before making a combined performance–energy claim.

## Recorded dependency environment

Observed in executed notebook outputs and JSON audit metadata, rather than inferred from the current machine:

| Component | Version recorded in this archive | Provenance |
|---|---|---|
| Python | **3.13.18** | D3 dataset-verification JSON |
| PyTorch | **2.11.0+cu130** | executed D3 notebooks |
| torchvision | **0.26.0+cu130** | executed ResNet50 and throughput notebooks |
| timm | **1.0.30** | executed D3 notebooks |
| CodeCarbon | **3.3.2** | installation command and executed D3 sustainability notebooks |
| kagglehub | **1.0.3** | executed D3 ResNet50 notebook |
| numpy | **2.1.3** | D3 dataset-verification JSON |
| pandas | **2.3.3** | D3 dataset-verification JSON |
| Pillow | **13.0.0** | D3 dataset-verification JSON (`PIL`) |

Other modules appear in notebook imports, but **their exact installed versions are not captured consistently** in the supplied archive: `scipy`, `scikit-learn`, `matplotlib`, `python-docx`, `scikit-image`, `opencv-python` (`cv2`), `transformers`, and `IPython`. `requirements.txt` therefore uses **declared compatibility ranges** for those entries instead of inventing an "actual" pinned version. For strict bitwise reproduction, export `python -m pip freeze > run-specific-freeze.txt` plus CUDA/GPU/driver details from each original Kaggle environment. Some notebooks themselves invoke `pip install` without version pins and may change the environment during execution.

**PyTorch wheel caution:** `torch==2.11.0` and `torchvision==0.26.0` in `requirements.txt` pin the recorded upstream versions but not the local `+cu130` CUDA build. Kaggle's existing GPU runtime or a matching platform-specific PyTorch wheel index is required to reproduce the CUDA builds and performance characteristics.

## Shared code, modules and reproducibility

- The **only standalone repository Python module** is [`REMNET/06_Dataset_Preparation/remnet_dataset_audit.py`](<REMNET/06_Dataset_Preparation/remnet_dataset_audit.py>). It uses standard-library imports, runs directly with the command above, and is **included** in the repository.
- Notebook-source inspection found **no imports of additional user-defined `.py` modules**. Training, evaluation and sustainability helper functions are defined inside the notebooks. External import dependencies are listed in `requirements.txt` (plus Python standard library).
- `config.yaml` records dataset links, the actual archive paths, split sizes, seeds and reporting conventions. It is provided for documentation/repository integration only; **notebooks do not parse it**. To change paths in existing notebooks, edit their own input-root/manifest variables.
- Data-source and per-seed sustainability reports are included, e.g. [`09_Sustainability_Results/REMNET_Sustainability_Five_Seed_Source_Data(1).csv`](<09_Sustainability_Results/REMNET_Sustainability_Five_Seed_Source_Data(1).csv>). These distinguish **CodeCarbon original** emissions from **standardized fixed-factor** CO₂e.
- No original image files or trained checkpoint **bytes** have been packaged here; only the user-provided screenshot directory scaffold has been added. The newly photographed Stage 1 manifests and nested `All models.zip` files are not included because their bytes were not supplied. Re-running frozen external tests requires compatible real weights, class order and manifests. Do **not** tune frozen models against V2 or D3 if claiming zero-shot evaluation.

## Citation and rights

Cite the original dataset and RealWaste as described in their linked source metadata. V2 source/citation information and unresolved licensing notes are preserved in its verification JSON. This repository's `LICENSE` deliberately **does not grant an open-source license** because no named copyright owner or permission to relicense the contributed code was provided. Dataset images, model weights, pretrained weights and third-party libraries are governed by **their own** terms and are **not** relicensed by this repository notice. Maintainers may replace the code-rights notice with a chosen SPDX-recognized license upon authorization from the rights holder(s).
