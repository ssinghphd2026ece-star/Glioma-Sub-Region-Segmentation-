# Claim traceability and decisions to confirm with manuscript authors

| Issue | Manuscript statement | Repository choice | Status |
|---|---|---|---|
| Data | BraTS2020 H5, four modalities, 369 people | H5 `image` and raw-label `mask`; patient-level 258/55/56 newly seeded split | Authors' split missing |
| Slices | 57,195 raw; 29,374 filtered; test 4,426 and also 8,680 full | Full ordered val/test volumes, filtered train only | Discrepancy in article |
| Normalization | Non-zero foreground, modality z-score | Independently per modality per axial H5 slice | Implementation inference |
| Image size | 240x240 -> 160x160 | 160 by default | Manuscript |
| Network | residual encoder/decoder, CBAM, gated skips, residual ASPP | One explicit PyTorch implementation | Exact block formulas/settings missing |
| Loss | Focal Tversky + Dice + weighted BCE | 0.7/0.3, 1.33, weights 1/1/1, pos_weight 3 | Numeric hyperparameters assumed |
| Training | AdamW LR 1e-4, batch 16 | AdamW, plateau scheduler, early stop | Weight decay and patiences assumed |
| Tuning | Validation-only region thresholds and component filtering | Coordinate search, grid 0.3..0.7 and 0/10/50/100 native voxels | Search grid/rules assumed |
| Metrics | Complete patient volumes, mean per patient, physical HD95 | 3D volumes with original H5 H/W and supplied voxel spacing | Requires verified original affine/orientation |
| Seeds | 42, 123, 256, 512, 1024 | Same five-seed list | Manuscript |
| Dice | 0.8982 in tables; 0.8915 elsewhere | Reports measured values only | Manuscript contradiction |
| Parameters | 10.90M, 24.8 GFLOPs | Measured by `scripts.profile` with selected THOP convention | Exact architecture not available |
| Cross dataset | BraTS2019 and BraTS2021 untuned | Separate test manifests, unchanged checkpoint/calibration | Subject overlap must be audited |

No original BraTS images, checkpoints, experiment logs, or manuscript-derived measured results are included. These files represent a runnable computational experiment framework rather than certification of reproduction of the paper's numeric findings.
