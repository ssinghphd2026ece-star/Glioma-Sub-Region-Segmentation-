"""Check manifest integrity, split leakage, continuity, dataset overlaps and masks."""
import argparse
from collections import defaultdict,Counter
from pathlib import Path
from src.experiment import read_manifest,save_json
from src.data import read_h5

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--external-manifest',action='append',default=[])
    p.add_argument('--out',default='audit.json');p.add_argument('--check-masks',action='store_true');a=p.parse_args()
    rows=read_manifest(a.manifest);by=defaultdict(list)
    for r in rows:by[r['patient_id']].append(r)
    violations=[]
    for pid,rs in by.items():
        splits={r['split'] for r in rs}
        if len(splits)>1:violations.append(f'Split leakage: {pid}')
        idx=sorted(int(r['slice_index']) for r in rs)
        if len(idx)!=len(set(idx)):violations.append(f'Duplicate slices: {pid}')
        if next(iter(splits))!='train' and idx!=list(range(idx[0],idx[0]+len(idx))):violations.append(f'Incomplete nontraining volume: {pid}')
        for r in rs:
            if not Path(r['path']).is_file():violations.append(f'Missing: {r["path"]}')
            elif a.check_masks:read_h5(r['path'])
    external={}
    for f in a.external_manifest:
        ext={r['patient_id'] for r in read_manifest(f)}
        external[f]=dict(patients=len(ext),exact_patient_id_overlap=sorted(ext&set(by)))
    report=dict(patient_count=len(by),slice_count=len(rows),split_patients=dict(Counter(next(iter({r['split'] for r in rs})) for rs in by.values())),
        split_slices=dict(Counter(r['split'] for r in rows)),violations=violations,external=external,
        caveat='Exact ID overlap is NOT sufficient to exclude re-released subjects across BraTS years; verify provenance separately.')
    save_json(a.out,report);print(report)
    if violations:raise SystemExit(1)
if __name__=='__main__':main()
