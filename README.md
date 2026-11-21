# REMNET: Waste Classification, Cross-Dataset Validation and Sustainable AI

**REMNET** is an ensemble image-classification research framework that combines **MobileNetV2**, **ResNet50**, and **EfficientNet-Lite0** for six-category waste recognition. This repository brings together the experimental notebooks, trained model checkpoints, dataset preparation evidence, independent external evaluations, statistical analyses, and CodeCarbon sustainability measurements.

The experiments use five seeds (`7`, `49`, `112`, `345`, `476`) and the common category order:

`cardboard · glass · metal · paper · plastic · trash`

**Repository:** [AkashKashyapGpu/Remnet_config_file](https://github.com/AkashKashyapGpu/Remnet_config_file)

## Research workflows

| Workflow | Purpose | Experimental arrangement |
|---|---|---|
| Original classification | Train and evaluate REMNET and the three standalone CNNs on the original waste-image dataset | Five-seed classification experiments |
| Sustainability | Measure energy use, CO₂e, and runtime with CodeCarbon |  Five-seed Sustainability experiments with training and inference |
| Garbage Classification V2 | Evaluate transfer on a screened six-category external cohort | Frozen classification checkpoints; 4,931 images |
| RealWaste D3 | Evaluate zero-shot performance on realistic external imagery | Frozen classification checkpoints; 3,497 images |
| Computational profiling | Measure architectural inference latency, throughput, and peak GPU memory | Randomly initialized architectures and a three-CNN ensemble; five profiling repetitions |

**Experiment provenance.** Classification and sustainability are two complementary, executed research workflows. Their checkpoints, measurements, and per-seed results are preserved under separate folders. Accuracy–energy indicators derived from their reported summaries describe cross-workflow comparisons. External classification uses the frozen classification checkpoint collection; sustainability reporting follows its own recorded monitoring sessions.

## Repository layout

The following paths match the folders currently available on GitHub. The directory name `chekpoints` is reproduced **exactly as stored in the repository** so that the links resolve correctly.

```text
Remnet_config_file/
├── README.md
├── requirements.txt
├── config.yaml
├── LICENSE
├── Checkpoints_trained_model/
│   └── chekpoints/
│       ├── Classification/
│       │   ├── Remnet/                    # 15 constituent weights: 3 CNNs × 5 seeds
│       │   ├── Resnet_50/                 # 5 standalone ResNet50 weights
│       │   ├── efficientnet_lite0/        # 5 standalone EfficientNet-Lite0 weights
│       │   └── mobilenet_v2/              # 5 standalone MobileNetV2 weights
│       └── Sustainability/
│           ├── Remnet/                    # 15 constituent weights: 3 CNNs × 5 seeds
│           ├── Resnet_50/                 # 5 standalone ResNet50 weights
│           ├── efficientnet_lite0/        # 5 standalone EfficientNet-Lite0 weights
│           └── mobilenet_v2/              # 5 standalone MobileNetV2 weights
├── REMNET/
│   ├── 01_Classification_Implementation/
│   ├── 02_Energy_and_Carbon_Measurement/
│   ├── 03_Statistical_Analysis/
│   ├── 04_Classification_Results/
│   ├── 05_Computational_Complexity/
│   └── 06_Dataset_Preparation/
├── 09_Sustainability_Results/
│   ├── CodeCarbon training and inference CSV logs/
│   └── REMNET_Sustainability_Five_Seed_Source_Data(1).csv
├── Garbage Classification V2 external validation/
│   └── 07_Garbage_Classification_V2/
│       ├── Dataset verification/
│       ├── Classification implementation/
│       └── Classification results/
└── RealWaste external validation/
    └── 08_RealWaste/
        ├── Dataset verification/
        ├── Classification implementation/
        ├── Classification results/
        └── Sustainability implementation/
```

### Trained checkpoint inventory

The repository hosts **60 `.pt` checkpoint files**, organized by experimental workflow and model role.

| Model role | Classification | Sustainability |
|---|---:|---:|
| REMNET constituent CNNs | 15 | 15 |
| Standalone ResNet50 | 5 | 5 |
| Standalone EfficientNet-Lite0 | 5 | 5 |
| Standalone MobileNetV2 | 5 | 5 |
| **Total** | **30** | **30** |

Browse the actual checkpoint folders: [Classification](Checkpoints_trained_model/chekpoints/Classification) · [Sustainability](Checkpoints_trained_model/chekpoints/Sustainability). Each five-seed set corresponds to seeds **7, 49, 112, 345, and 476**. The REMNET directories contain three constituent-model files per seed; the ensemble combines their outputs during inference.

## Data sources and evaluation cohorts

| Role | Dataset and source | Image cohort | Local/Kaggle mounting guidance |
|---|---|---:|---|
| Original D1 | [Garbage Classification Image Dataset](https://www.kaggle.com/datasets/saumyamohandas/garbage-classification-image-dataset) | Original experimental evaluation: 463 images | `/kaggle/input/datasets/saumyamohandas/garbage-classification-image-dataset` |
| External V2 | [Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2), dataset version 12 | 4,931 screened images | `/kaggle/input/datasets/sumn2u/garbage-classification-v2` |
| External D3 | [RealWaste — UCI repository](https://archive.ics.uci.edu/dataset/908/realwaste) ([DOI](https://doi.org/10.24432/C5SS4G)); [Kaggle mirror](https://www.kaggle.com/datasets/akash264208/real-waste) | 3,497 screened images | `/kaggle/input/datasets/akash264208/real-waste` |

The dataset verification directories provide class mappings, source metadata, image manifests, and screening records. For D3, **Miscellaneous Trash** maps to the model's **trash** category. Use the saved manifest belonging to each external dataset to retain the same evaluation cohort across models and seeds.

### Stage 1 split documentation

The [`REMNET/06_Dataset_Preparation/`](REMNET/06_Dataset_Preparation) directory contains the locked corrected-dataset preparation records:

- [`REMNET_Final_Retained_Split_Manifest.csv`](REMNET/06_Dataset_Preparation/REMNET_Final_Retained_Split_Manifest.csv) and [`REMNET_All_2906_Locked_Dataset_Splits.csv`](REMNET/06_Dataset_Preparation/REMNET_All_2906_Locked_Dataset_Splits.csv) — retained image assignments and identifiers.
- [`REMNET_Training_Split.csv`](REMNET/06_Dataset_Preparation/REMNET_Training_Split.csv) — **1,956** training entries.
- [`REMNET_Validation_Split.csv`](REMNET/06_Dataset_Preparation/REMNET_Validation_Split.csv) — **487** validation entries.
- [`REMNET_Testing_Split.csv`](REMNET/06_Dataset_Preparation/REMNET_Testing_Split.csv) — **463** testing entries.
- [`REMNET_Classwise_Partition_Counts.csv`](REMNET/06_Dataset_Preparation/REMNET_Classwise_Partition_Counts.csv) and [`REMNET_Split_Export_Verification.json`](REMNET/06_Dataset_Preparation/REMNET_Split_Export_Verification.json) — class-level distributions and split checks.
- [`README_Stage1_Dataset_Split_Evidence.txt`](REMNET/06_Dataset_Preparation/README_Stage1_Dataset_Split_Evidence.txt) — original split provenance and verification procedure.

These exported Stage 1 manifests document a **2,906-image corrected dataset**. Archived experimental notebooks and separate audit exports maintain their original, workflow-specific cohort definitions; use the associated manifest to reproduce each reported experiment.

## Environment and installation

The notebook outputs record the following core software environment:

| Component | Recorded version |
|---|---|
| Python | 3.13.18 |
| PyTorch | 2.11.0+cu130 |
| torchvision | 0.26.0+cu130 |
| timm | 1.0.30 |
| CodeCarbon | 3.3.2 |
| KaggleHub | 1.0.3 |
| NumPy | 2.1.3 |
| pandas | 2.3.3 |
| Pillow | 13.0.0 |

[`requirements.txt`](requirements.txt) pins the recorded release versions and supplies compatibility ranges for other imported packages. Install in the intended Python and CUDA environment:

```bash
git clone https://github.com/AkashKashyapGpu/Remnet_config_file.git
cd Remnet_config_file
python -m pip install -r requirements.txt
```

On Kaggle, enable a GPU for training, inference, and benchmarking; attach the named datasets through **Add Input**, then open and run the selected notebook from its first cell. Notebook-generated reports are ordinarily exported to `/kaggle/working/`. CUDA wheel variants depend on the runtime, so select a GPU environment compatible with the recorded PyTorch and torchvision builds.

[`config.yaml`](config.yaml) provides a consolidated **reference** for dataset locations, class ordering, checkpoint mounts, seeds, image preprocessing, and sustainability settings. Each archived notebook retains its own internal configuration variables; set those paths to the corresponding attached Kaggle datasets or local directories when running a notebook in a new environment.

### Checkpoint input paths

The repository's Git-tracked checkpoint collection is located at:

`Checkpoints_trained_model/chekpoints/Classification/`  
`Checkpoints_trained_model/chekpoints/Sustainability/`

Existing external-evaluation notebooks were written for specific Kaggle dataset mounts, including:

- Classification: `/kaggle/input/datasets/akash264208/checkpoint-manifist`
- D3 monitored sustainability: `/kaggle/input/datasets/akash264208/manifist-part2/Section3`

Attach the required checkpoint/manifest dataset or set the notebook's input-root variables to the equivalent file organization. Preserve the existing trained weights and class ordering throughout zero-shot evaluation.

## Running the notebooks

Every notebook is a self-contained research workflow. Open it in Kaggle/Jupyter, attach the relevant input data, set paths for the current environment, and **Run all**. Results should be retained with their model, seed, cohort, and experimental-workflow labels.

### A. Original classification and sustainability

| Notebook | Purpose |
|---|---|
| [`Efficientnet_lite0_accu.ipynb`](REMNET/01_Classification_Implementation/Efficientnet_lite0_accu.ipynb) | Original-dataset EfficientNet-Lite0 classification. |
| [`MobileNet_Accu_s2.ipynb`](REMNET/01_Classification_Implementation/MobileNet_Accu_s2.ipynb) | Original-dataset MobileNetV2 classification. |
| [`Resnet_50_Acc_s2.ipynb`](REMNET/01_Classification_Implementation/Resnet_50_Acc_s2.ipynb) | Original-dataset ResNet50 classification. |
| [`Remnet_Accu_s2.ipynb`](REMNET/01_Classification_Implementation/Remnet_Accu_s2.ipynb) | REMNET constituent training and ensemble classification. |
| [`efficientnet-lite-prob-co2-s2.ipynb`](REMNET/02_Energy_and_Carbon_Measurement/efficientnet-lite-prob-co2-s2.ipynb) | Independent EfficientNet-Lite0 sustainability experiment. |
| [`mobilenet-v2-prob-c02-s2.ipynb`](REMNET/02_Energy_and_Carbon_Measurement/mobilenet-v2-prob-c02-s2.ipynb) | Independent MobileNetV2 sustainability experiment. |
| [`resnet-50-prob-co2-s2.ipynb`](REMNET/02_Energy_and_Carbon_Measurement/resnet-50-prob-co2-s2.ipynb) | Independent ResNet50 sustainability experiment. |
| [`remnet-co2-prob-s2.ipynb`](REMNET/02_Energy_and_Carbon_Measurement/remnet-co2-prob-s2.ipynb) | Independent REMNET training-energy and ensemble-inference monitoring. |
| [`REMNET_Classification_Paired_T_Test_Stage2.ipynb`](REMNET/03_Statistical_Analysis/REMNET_Classification_Paired_T_Test_Stage2.ipynb) | Statistical comparison of the original classification results. |
| [`REMNET_Sustainability_Paired_T_Test_Stage2.ipynb`](REMNET/03_Statistical_Analysis/REMNET_Sustainability_Paired_T_Test_Stage2.ipynb) | Statistical comparison of sustainability measurements. |
| [`Gflops.ipynb`](REMNET/05_Computational_Complexity/Gflops.ipynb) | Model computational-complexity profiling. |
| [`throughput.ipynb`](REMNET/05_Computational_Complexity/throughput.ipynb) | Architecture-level inference benchmark reported as **Table 15** in the manuscript; detailed protocol below. |

Original Stage 2 classification and CodeCarbon notebook preprocessing uses **32 × 32** image inputs. Sustainability reports supports original CodeCarbon CO₂e values from calculated values based on the stated carbon-intensity factor of **0.475 kg CO₂e/kWh**.

### B. Garbage Classification V2 — external evaluation

Run the dataset preparation notebooks first, then the four frozen-checkpoint classification notebooks, followed by statistical analysis:

| Order | Notebook | Action |
|---:|---|---|
| 1 | [`stage2_part1.ipynb`](<Garbage Classification V2 external validation/07_Garbage_Classification_V2/Dataset verification/stage2_part1.ipynb>) | Source discovery and inventory. |
| 2 | [`s3p2.ipynb`](<Garbage Classification V2 external validation/07_Garbage_Classification_V2/Dataset verification/s3p2.ipynb>) | Six-class mapping, overlap screening, and retained manifest verification. |
| 3 | [`efficientnetlite0.ipynb`](<Garbage Classification V2 external validation/07_Garbage_Classification_V2/Classification implementation/efficientnetlite0.ipynb>) | Frozen EfficientNet-Lite0 inference. |
| 4 | [`mobilenet-v2.ipynb`](<Garbage Classification V2 external validation/07_Garbage_Classification_V2/Classification implementation/mobilenet-v2.ipynb>) | Frozen MobileNetV2 inference. |
| 5 | [`resnet-50.ipynb`](<Garbage Classification V2 external validation/07_Garbage_Classification_V2/Classification implementation/resnet-50.ipynb>) | Frozen ResNet50 inference. |
| 6 | [`remnet123.ipynb`](<Garbage Classification V2 external validation/07_Garbage_Classification_V2/Classification implementation/remnet123.ipynb>) | Frozen REMNET ensemble inference. |
| 7 | [`classification-test.ipynb`](<Garbage Classification V2 external validation/07_Garbage_Classification_V2/Classification implementation/classification-test.ipynb>) | Statistical analysis of saved five-seed V2 results. |

**Outputs:** [`Classification results/`](<Garbage Classification V2 external validation/07_Garbage_Classification_V2/Classification results/>) includes REMNET five-seed CSV summaries and [`All models.zip`](<Garbage Classification V2 external validation/07_Garbage_Classification_V2/Classification results/All models.zip>), containing consolidated model outputs.

### C. RealWaste D3 — external classification and sustainability

First verify the D3 cohort; then run the frozen classification evaluations and, separately, the monitored sustainability notebooks.

| Order | Notebook | Action |
|---:|---|---|
| 1 | [`d3-s2.ipynb`](<RealWaste external validation/08_RealWaste/Dataset verification/d3-s2.ipynb>) | Validate D3 source, mapping, screened manifest, and cohort identity. |
| 2 | [`efficientnetlite0-class-d3.ipynb`](<RealWaste external validation/08_RealWaste/Classification implementation/efficientnetlite0-class-d3.ipynb>) | Frozen EfficientNet-Lite0 classification. |
| 3 | [`mobilenet-class-d3.ipynb`](<RealWaste external validation/08_RealWaste/Classification implementation/mobilenet-class-d3.ipynb>) | Frozen MobileNetV2 classification. |
| 4 | [`resnet50-class-d3.ipynb`](<RealWaste external validation/08_RealWaste/Classification implementation/resnet50-class-d3.ipynb>) | Frozen ResNet50 classification. |
| 5 | [`remnet-class-d3.ipynb`](<RealWaste external validation/08_RealWaste/Classification implementation/remnet-class-d3.ipynb>) | Frozen REMNET ensemble classification. |
| 6 | [`class_test.ipynb`](<RealWaste external validation/08_RealWaste/Classification implementation/class_test.ipynb>) | Statistical analysis of the five-seed D3 classification records. |
| 7 | [`efficient-co2-d3.ipynb`](<RealWaste external validation/08_RealWaste/Sustainability implementation/efficient-co2-d3.ipynb>) | Separately monitored EfficientNet-Lite0 inference. |
| 8 | [`mobilenetv2-co2-d3.ipynb`](<RealWaste external validation/08_RealWaste/Sustainability implementation/mobilenetv2-co2-d3.ipynb>) | Separately monitored MobileNetV2 inference. |
| 9 | [`resnet-50-co2-d3.ipynb`](<RealWaste external validation/08_RealWaste/Sustainability implementation/resnet-50-co2-d3.ipynb>) | Separately monitored ResNet50 inference. |
| 10 | [`remnet-sustainable-d3.ipynb`](<RealWaste external validation/08_RealWaste/Sustainability implementation/remnet-sustainable-d3.ipynb>) | Separately monitored REMNET inference. |

**Outputs:** [`Classification results/`](<RealWaste external validation/08_RealWaste/Classification results/>) contains [`REMNET_All_Seeds_D3_External.csv`](<RealWaste external validation/08_RealWaste/Classification results/REMNET_All_Seeds_D3_External.csv>), [`REMNET_Five_Seed_Mean_SD_D3_External.csv`](<RealWaste external validation/08_RealWaste/Classification results/REMNET_Five_Seed_Mean_SD_D3_External.csv>), and [`all models.zip`](<RealWaste external validation/08_RealWaste/Classification results/all models.zip>).

D3 monitoring records **two inference passes** over the same 3,497 images, corresponding to **6,994 processed images per seed**. Inference energy, runtime, throughput, and CO₂e are retained as sustainability-specific outcomes.

## Manuscript architecture-level GPU inference benchmark

[`REMNET/05_Computational_Complexity/throughput.ipynb`](REMNET/05_Computational_Complexity/throughput.ipynb) is an **architecture-level inference benchmark**. It instantiates the MobileNetV2, EfficientNet-Lite0, and ResNet50 architectures with **randomly initialized weights** and evaluates their three-model REMNET soft-voting ensemble. The benchmark **uses no trained classification checkpoints** and is designed to characterize architectural inference cost separately from trained-model accuracy and separately monitored energy measurements.

The notebook measures:

- **Batch-size-1:** synchronized single-image forward-pass latency (mean and p95).
- **Batch-size-32:** throughput (images per second) and peak GPU-allocated memory.
- **Input:** 32 × 32 RGB images under the notebook's defined preprocessing.
- **Repeatability:** five profiling repetitions (rather than five training seeds).
- **Real-time criterion:** inference-only batch-1 p95 ≤ 33.33 ms for the tested GPU and resolution.

 Classification results, independent CodeCarbon measurements, and architecture-only profiling each retain their respective measurement scopes.

## Shared modules and result provenance

The research notebooks define their workflow helpers within notebook cells and use third-party libraries listed in [`requirements.txt`](requirements.txt). No additional repository-local Python package import is required by the archived notebook workflows.

Five-seed source records, training/inference logs, and standardized-versus-native carbon measurements are maintained in [`09_Sustainability_Results/`](09_Sustainability_Results), including [`REMNET_Sustainability_Five_Seed_Source_Data(1).csv`](<09_Sustainability_Results/REMNET_Sustainability_Five_Seed_Source_Data(1).csv>).

For reproducible comparisons, retain the original seed, dataset manifest, checkpoint identity, class order, preprocessing settings, and notebook outputs for each experiment. The classification and sustainability checkpoint folders preserve their corresponding independent experimental histories.

## Citation and permissions

When reusing this work, cite the associated REMNET manuscript and the original dataset providers. Dataset source URLs, license metadata, and citation information are included in the V2 and D3 verification directories. Refer to [`LICENSE`](LICENSE) for the repository's code-rights notice; images, pretrained weights, and external packages remain governed by their respective original providers.
