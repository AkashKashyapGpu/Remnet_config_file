REMNET — Stage 1: Frozen corrected image-level dataset split evidence

SOURCE OF TRUTH:
The image assignments are direct projections from REMNET_Final_Retained_Split_Manifest.csv, SHA-256:
75457c8836ccfd269756b9f1f1db92bf737b81fe122def35fcb2adea50fd271c
No images were resampled, relabeled, or repartitioned during this export.

FILES:
- REMNET_Training_Split.csv: 1,956 rows
- REMNET_Validation_Split.csv: 487 rows
- REMNET_Testing_Split.csv: 463 rows
- REMNET_All_2906_Locked_Dataset_Splits.csv: 2,906 rows
- REMNET_Classwise_Partition_Counts.csv: distribution of all six waste classes
- REMNET_Final_Retained_Split_Manifest.csv: unchanged locked audit master

COLUMNS IN SPLIT CSV:
image_filename: file basename, e.g. cardboard337.jpg
class_label: cardboard / glass / metal / paper / plastic / trash
partition: Training / Validation / Testing
relative_image_path: file path relative to REMNET_Corrected_Dataset_and_Protocol.zip, including dataset_clean/
sha256: SHA-256 digest of the image's bytes from the verified master manifest
original_source_path: original source dataset location

VERIFICATION:
Split counts and class-wise totals match the locked source manifest. Every final_path was checked as a unique image member inside REMNET_Corrected_Dataset_and_Protocol.zip. Original and final hash verification flags in the master manifest were MATCH. The Stage 1 dataset audit previously verified all retained image hashes against source and cleaned files. No GPU execution is performed by CSV extraction.

IMPORTANT SCIENTIFIC PROVENANCE:
Historical classification and CodeCarbon experiments used 2,508 training images and 454 testing images, without validation. These CSVs instead describe the CLEANED and VERIFIED dataset intended for Stage 2 experiments (Training 1,956 / Validation 487 / Testing 463). They do NOT imply Stage 2 experiments have been run.

SPLIT POLICY:
Fixed approximately stratified 80/20 Training/Validation split of deduplicated original training, with random_state=42 and subsequent documented visually verified exclusions. Testing is an independently retained corrected original test split, not a fresh 80/20/20 redistribution. Freeze image memberships from the provided manifest rather than recomputing the split.
Six conflicted-label images remain quarantined and four confirmed visual duplicates were excluded; 13 remaining dHash <=2 cross-split candidates were visually reviewed and retained as distinct images.
