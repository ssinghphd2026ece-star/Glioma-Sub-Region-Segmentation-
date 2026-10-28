"""Shared reproducibility utilities for experiment scripts."""
import csv,json,random
from pathlib import Path
import numpy as np
import torch
from src.model import GliomaUNet


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def read_manifest(path,split=None):
    with open(path,newline='') as f:rows=list(csv.DictReader(f))
    required={'path','patient_id','slice_index','split'}
    if rows and not required.issubset(rows[0]):raise ValueError(f'Manifest missing {required-set(rows[0])}')
    root=Path(path).resolve().parent
    for r in rows:
        p=Path(r['path']);r['path']=str(p if p.is_absolute() else root/p)
    return [r for r in rows if split is None or r['split']==split]

def load_checkpoint(path,device):
    # weights_only=False: checkpoints should be trusted files created by this project.
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    m=GliomaUNet(aspp=bool(ckpt['aspp']),cbam=bool(ckpt['cbam']),gates=bool(ckpt['gates']))
    m.load_state_dict(ckpt['model']);return m.to(device).eval(),ckpt

def get_device(name='auto'):
    return torch.device('cuda' if (name=='auto' and torch.cuda.is_available()) else ('cpu' if name=='auto' else name))

def save_json(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False,default=lambda x:float(x)))

def save_csv(path,rows):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
