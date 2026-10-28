import argparse,csv,random
from pathlib import Path
from collections import defaultdict
from src.data import patient_id,slice_index,read_h5
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',default='manifest.csv');p.add_argument('--seed',type=int,default=42);p.add_argument('--train-background-keep',type=float,default=.3);a=p.parse_args()
files=sorted(Path(a.data).rglob('*.h5'));groups=defaultdict(list)
for f in files:groups[patient_id(f)].append(f)
patients=sorted(groups);random.Random(a.seed).shuffle(patients)
if len(patients)!=369:print(f'WARNING: {len(patients)} patients; manuscript describes 369')
train=set(patients[:258]);val=set(patients[258:313]);test=set(patients[313:369]);rng=random.Random(a.seed)
with open(a.out,'w',newline='') as fh:
 w=csv.DictWriter(fh,fieldnames=['path','patient_id','slice_index','split','has_tumor']);w.writeheader()
 for pid in patients:
  split='train' if pid in train else 'val' if pid in val else 'test'
  for f in sorted(groups[pid],key=slice_index):
   _,mask=read_h5(f);tumor=int((mask!=0).any())
   if split=='train' and not tumor and rng.random()>a.train_background_keep:continue
   w.writerow(dict(path=str(f.resolve()),patient_id=pid,slice_index=slice_index(f),split=split,has_tumor=tumor))
print('Wrote',a.out,'; split is newly generated, not the authors original split')
