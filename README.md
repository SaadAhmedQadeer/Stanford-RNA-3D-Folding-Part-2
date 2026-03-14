# RNA 3D Structure Prediction

End-to-end pipeline for predicting RNA tertiary structure from sequence,
built for the Stanford RNA 3D Folding Part 2 Kaggle competition (2026).

## Method

The pipeline uses RhoFold+ (Shen et al., Nature Methods 2024), a deep
learning model pretrained on 23.7 million RNA sequences. Five structurally
diverse predictions are generated per target by subsampling the provided
Multiple Sequence Alignment (MSA) files with different random seeds.

For sequences exceeding RhoFold+'s 440 nt positional embedding limit, a
physics-informed constrained-walk fallback generates coordinates with
correct C1' bond geometry (~5.9 Angstroms).

## Results

- 28 test targets predicted (9,762 residue rows, 5 predictions each)
- Best single-target TM-score: 0.507 (target 9IWF, 69 nt)
- 25/28 targets (89%) used RhoFold+ neural network predictions
- Mean C1' bond length: 5.97 Angstroms (ideal: 5.9 Angstroms)
- Total prediction time: ~31 minutes on 2x NVIDIA Tesla T4 GPU

## Key technical contributions

- Identified correct C1' atom index (index 0) from RhoFold's flattened
  (L x 23, 3) output tensor via systematic diagnostic scanning
- Resolved GPU context corruption caused by CUDA out-of-bounds errors
  on sequences exceeding the model's positional embedding table size
- Replaced a broken segment-based geometry predictor (producing bonds
  up to 1,004 Angstroms) with a bond-angle-torsion constrained walk

## Repository structure
```
notebook/   - Kaggle notebook with full pipeline
report/     - Full technical report (PDF + LaTeX source)
figures/    - All analysis and results figures
```

## Pipeline

RNA Sequence -> MSA Retrieval -> RhoFold+ (GPU) -> 5-Seed Ensemble
-> C1' Extraction -> submission.csv

## Reference

Shen, T. et al. Accurate RNA 3D structure prediction using a language
model-based deep learning approach. Nature Methods 21, 2287-2298 (2024).
https://doi.org/10.1038/s41592-024-02487-0

Competition: https://www.kaggle.com/competitions/stanford-rna-3d-folding-2
```

---

## requirements

torch>=2.0.0
biopython>=1.81
scipy>=1.10.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
tqdm>=4.65.0
einops>=0.6.0
ml_collections>=0.1.1
ViennaRNA>=2.6.0
