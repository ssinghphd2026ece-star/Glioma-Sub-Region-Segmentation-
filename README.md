Role-Separated Dual-Attention Residual ASPP U-Net — Complete Reproduction Pipeline

1. Install

Python 3.10+ recommended; CUDA GPU recommended for real BraTS analysis.

```bash
python -m venv .venv
source .venv/bin/activate         
pip install -r requirements.txt
python -m pytest -q
```

2. Prepare BraTS2020 H5 dataset

```bash
python -m scripts.prepare_manifest --data /path/to/BraTS2020_H5 --out manifest.csv --seed 42 --train-background-keep 0.3
python -m scripts.audit_data --manifest manifest.csv --check-masks --out data_audit.json
```
For other cohorts, supply a CSV manifest with `path,patient_id,slice_index,split,has_tumor`; use `split=test` for external evaluations. Cross-year subject identity leakage must be investigated using original cohort provenance; names alone are insufficient.


```json
{
  "BraTS20_Training_001": {"spacing_zyx": [1.0, 1.0, 1.0]}
}
```
3. Train the proposed architecture

```bash
python -m scripts.train --manifest manifest.csv --config configs/default.json --out runs/full --seed 42
```
4. Optimize validation thresholds and components 

```bash
python -m scripts.calibrate --manifest manifest.csv --checkpoint runs/full/best.pt --out runs/full/calibration.json
```
5. Evaluate previously unseen 2020 test subjects

```bash
python -m scripts.evaluate --checkpoint runs/full/best.pt --manifest manifest.csv --split test --calibration runs/full/calibration.json --geometry geometry.json --out runs/full/evaluation --save-predictions
```

Outputs: `per_patient.csv` (WT/TC/ET metrics per patient), `summary.json` and, optionally, compressed prediction arrays. Uses original H5 mask dimensions and complete axial sequences, with native voxel spacings for physical HD95. Unilateral empty region HD95 is `NaN` (excluded from mean, valid count shown); double-empty HD95 is 0. Dice for double empty=1, unilateral empty=0.

```bash
python -m scripts.evaluate --checkpoint runs/full/best.pt --manifest manifest.csv --split test --calibration runs/full/calibration.json --skip-hd95 --out runs/full/evaluation
```

6. Eight factorial ablations (Table 5)

```bash
python -m scripts.ablation --manifest manifest.csv --config configs/default.json
```

7. Complete end-to-end ablations, five-seed repeat, external datasets

```bash
python -m scripts.run_all --manifest manifest.csv --geometry geometry.json --out runs/ablation --config configs/default.json
python -m scripts.repeat_seeds --manifest manifest.csv --geometry geometry.json --out runs/seeds
```

External BraTS2019 or BraTS2021:create a CSV manifest for each dataset, with `split=test`, and use the same BraTS2020 checkpoint and calibration.json, without fine tuning:

```bash
python -m scripts.evaluate --checkpoint runs/full/best.pt --manifest brats2019_manifest.csv --split test --calibration runs/full/calibration.json --geometry brats2019_geometry.json --out runs/external2019
python -m scripts.evaluate --checkpoint runs/full/best.pt --manifest brats2021_manifest.csv --split test --calibration runs/full/calibration.json --geometry brats2021_geometry.json --out runs/external2021
```

Patient ID overlap checks:

```bash
python -m scripts.audit_data --manifest manifest.csv --external-manifest brats2019_manifest.csv --external-manifest brats2021_manifest.csv
```

License and citation

Cite the supplied manuscript and the relevant BraTS dataset papers when publishing results. Obtain necessary permissions for dataset use. 
