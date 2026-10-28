# Role-Separated Dual-Attention Residual ASPP U-Net — Complete Reproduction Pipeline

**Source:** user-supplied *Role-Separated Dual-Attention Residual ASPP U-Net for Efficient 2D Glioma Sub-Region Segmentation* (SPIE formatted manuscript).

This is an **independent reconstruction**, not the original author code, and does not contain BraTS data, original trained weights, or fabricated measurements. The code supports preparation, train, 3D validation-only calibration, full-volume held-out evaluation, physical HD95, external datasets, eight factorial ablations, five seeds, inference, diagrams/figures via results, data audits, and profiling.

## 1. Install

Python 3.10+ recommended; CUDA GPU recommended for real BraTS experiments.

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
```

## 2. Prepare BraTS2020 H5 dataset

Expected H5 keys: `image` with `(240,240,4)` (or `(4,240,240)`) and **raw integer** `mask` with `(240,240)` and label set `0,1,2,4`. Channel order must be **T1, T1ce, T2, FLAIR**: check the provenance of your particular H5 conversion (many public conversions have other channel orders). The supplied manuscript lists these modalities but does not definitively identify the stored H5 channel order. Incorrect channel order invalidates replication. Filenames must have explicit patient and slice indices (`BraTS20_Training_001_slice_042.h5`), or adapt `src/data.py`.

```bash
python -m scripts.prepare_manifest --data /path/to/BraTS2020_H5 --out manifest.csv --seed 42 --train-background-keep 0.3
python -m scripts.audit_data --manifest manifest.csv --check-masks --out data_audit.json
```

The manifest generator makes a **new** patient-wise partition (258/55/56 for the manuscript's 369-person version). Train-only background subsampling is performed, while validation/test preserve full patient volumes. **Original author's exact patient IDs and background selection fraction are unavailable.** The paper inconsistently reports 4,426 and 8,680 test slices. The complete 56×155=8,680 slice count is required for unfiltered testing. Manifest generation requires 369 patients; review an alternate H5 edition before applying its split.

For other cohorts, supply a CSV manifest with `path,patient_id,slice_index,split,has_tumor`; use `split=test` for external evaluations. Cross-year subject identity leakage must be investigated using original cohort provenance; names alone are insufficient.

### Physical spacing for HD95

An H5 slice may not retain NIfTI header spacing. Provide **actual per-patient physical spacing** in z,y,x order (mm). Example for illustration only (do not treat this as verified image geometry):

```json
{
  "BraTS20_Training_001": {"spacing_zyx": [1.0, 1.0, 1.0]}
}
```

Save as `geometry.json` after extracting spacing from the **corresponding original NIfTI headers**, respecting array orientation. Otherwise use `--skip-hd95`; never report fictitious millimetre distances.

## 3. Train the proposed architecture

```bash
python -m scripts.train --manifest manifest.csv --config configs/default.json --out runs/full --seed 42
```

Outputs: `runs/full/best.pt`, `history.csv`, `run_config.json`. This uses a fixed threshold 0.5 for training-time checkpoint selection on validation slices, then full-volume validation-only calibration as a separate stage. This checkpoint rule is a reconstruction assumption, not proof of matching the manuscript's criterion.

## 4. Optimize validation thresholds and components (NO test data)

```bash
python -m scripts.calibrate --manifest manifest.csv --checkpoint runs/full/best.pt --out runs/full/calibration.json
```

Uses region-wise coordinate search maximizing patient-wise **3D Dice** on the complete validation set. Defaults: thresholds `0.3,0.4,0.5,0.6,0.7`, minimum connected-component sizes `0,10,50,100` native-voxel units, 3D 26-connectivity. These settings and the search method are explicit **assumptions**, because the paper does not disclose them. Connected components are removed before hierarchy restoration.

## 5. Evaluate previously unseen 2020 test subjects

```bash
python -m scripts.evaluate --checkpoint runs/full/best.pt --manifest manifest.csv --split test --calibration runs/full/calibration.json --geometry geometry.json --out runs/full/evaluation --save-predictions
```

Outputs: `per_patient.csv` (WT/TC/ET metrics per patient), `summary.json` and, optionally, compressed prediction arrays. Uses original H5 mask dimensions and complete axial sequences, with native voxel spacings for physical HD95. Unilateral empty region HD95 is `NaN` (excluded from mean, valid count shown); double-empty HD95 is 0. Dice for double empty=1, unilateral empty=0.

If physical image metadata is unavailable:

```bash
python -m scripts.evaluate --checkpoint runs/full/best.pt --manifest manifest.csv --split test --calibration runs/full/calibration.json --skip-hd95 --out runs/full/evaluation
```

## 6. Eight factorial ablations (Table 5)

```bash
python -m scripts.ablation --manifest manifest.csv --config configs/default.json
```

For every run folder (`runs/unet`, `runs/aspp`, `runs/cbam`, `runs/ag`, `runs/aspp_cbam`, `runs/aspp_ag`, `runs/cbam_ag`, `runs/full`), run calibration and evaluation separately using commands above (or use `scripts/run_all.py` below). All ablations reuse the same patient split and hyperparameters.

## 7. Complete end-to-end ablations, five-seed repeat, external datasets

```bash
python -m scripts.run_all --manifest manifest.csv --geometry geometry.json --out runs/ablation --config configs/default.json
python -m scripts.repeat_seeds --manifest manifest.csv --geometry geometry.json --out runs/seeds
```

**External BraTS2019 or BraTS2021:** create a CSV manifest for each dataset, with `split=test`, and use *the same BraTS2020 checkpoint and calibration.json*, without fine tuning:

```bash
python -m scripts.evaluate --checkpoint runs/full/best.pt --manifest brats2019_manifest.csv --split test --calibration runs/full/calibration.json --geometry brats2019_geometry.json --out runs/external2019
python -m scripts.evaluate --checkpoint runs/full/best.pt --manifest brats2021_manifest.csv --split test --calibration runs/full/calibration.json --geometry brats2021_geometry.json --out runs/external2021
```

Patient ID overlap checks:

```bash
python -m scripts.audit_data --manifest manifest.csv --external-manifest brats2019_manifest.csv --external-manifest brats2021_manifest.csv
```

## 8. Individual predicted masks, charts, and model complexity

```bash
python -m scripts.predict --manifest manifest.csv --patient BraTS20_Training_001 --checkpoint runs/full/best.pt --calibration runs/full/calibration.json --slice 80 --out figures/patient_001.npz
python -m scripts.plot_results --runs runs/ablation --out figures
python -m scripts.profile
```

The per-patient `.npz` prediction is `(3,Z,H,W)`, ordered WT, TC, ET. `plot_results` builds charts from measured `history.csv` and `evaluation/summary.json`; never reconstruct publication figures by inventing data points.

## 9. Synthetic smoke test without BraTS dataset

```bash
python -m scripts.create_synthetic_data --out demo_data --slices 3
python -m scripts.audit_data --manifest demo_data/manifest.csv --check-masks
```

The synthetic data are **not** representative of genuine MRI, and cannot verify clinical or paper performance. For tiny demo volumes, use `--allow-incomplete-volumes` on calibration and evaluation. Full 512-filter architecture is intended for GPU training.

## Architecture and training choices

Four-channel input -> five encoder scales (`32,64,128,256,512`) with residual conv blocks and optional CBAM -> residual ASPP bottleneck with dilation rates `(1,2,4,8)` plus global-pooling branch -> attention-gated skip transfer -> residual block plus CBAM at each decoder scale -> three sigmoid channels (implemented as raw logits plus sigmoid in loss/inference). Targets: WT=`{1,2,4}`, TC=`{1,4}`, ET=`{4}`; three independent binary outputs. Dataset normalization is per-slice, per-modality, nonzero-foreground z-score; images are bilinear-resized, masks nearest-neighbour resized, and training receives flip, 90° rotations, intensity scaling, shifts and noise.

The default loss is differentiable Focal Tversky + Dice + positive-weighted BCE. AdamW, LR=1e-4, batch 16, ReduceLROnPlateau, early stopping and mixed precision follow the manuscript in broad terms. All unspecified numeric choices are documented in `configs/default.json`.

## Reproducibility caveats and manuscript inconsistencies

- **No guarantee of published results:** reported WT/TC/ET Dice `0.9178/0.8912/0.8856`, mean `0.8982`, parameter count `10.90M`, compute `24.8 GFLOPs` must all be **measured** in the recreated environment. This architecture is one documented reconstruction, not a verified byte-for-byte original.
- Manuscript reports mean Dice `0.8915` in running text vs `0.8982` in tables; ablation claims `0.8265` vs tabulated `0.8150`; and 4,426 vs full 8,680 internal-test slices.
- Author did not provide exact random split, all loss weights, background filtering proportion, residual/ASPP/CBAM channel details, exact postprocessing search grid, training machine/software, or trained checkpoints. Exact replication requires original logs and configuration.
- BraTS year releases can include the same subjects. The term *cross-dataset* does not itself guarantee independent patients.
- HD95 implementation uses symmetric max of directional 95th-percentile surface distances; implementations of surface weighting and empty-region policy may differ from challenge scoring.
- Patient-level macro-average metrics are reported; background-influenced accuracy and specificity are secondary.
- **Research only. Not for clinical diagnosis or treatment decisions.** Obtain data from official sources under applicable license; do not commit patient MRI scans.

## Repository map

- `src/model.py`: U-Net, residual blocks, CBAM, residual ASPP, attention gates.
- `src/data.py`: H5 reading, BraTS target preparation, augmentation.
- `src/losses.py`: composite differentiable optimization objective.
- `src/volumes.py`: patient grouping, geometry, native-size volume reconstruction.
- `src/evaluation.py`: connected components, hierarchy, volumetric metrics/HD95.
- `src/experiment.py`: shared I/O, loading, random seeds.
- `scripts/prepare_manifest.py`, `scripts/audit_data.py`: splitting and audit.
- `scripts/train.py`, `scripts/calibrate.py`, `scripts/evaluate.py`, `scripts/predict.py`: full pipeline.
- `scripts/ablation.py`, `scripts/run_all.py`, `scripts/repeat_seeds.py`: experiments.
- `scripts/plot_results.py`, `scripts/profile.py`: plots and complexity.
- `scripts/create_synthetic_data.py`: smoke-test dataset.
- `tests/`: unit tests.

## License and citation

Cite the supplied manuscript and the relevant BraTS dataset papers when publishing results. Obtain necessary permissions for dataset use. The generated implementation contains no redistribution of dataset images or author checkpoints.
