"""Volume assembly and physical geometry helpers; axes are (Z,H,W)."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data import H5Slices, read_h5


def group_patients(records):
    groups=defaultdict(list)
    for r in records: groups[r['patient_id']].append(r)
    for patient in groups:
        groups[patient].sort(key=lambda r:int(r['slice_index']))
        indexes=[int(r['slice_index']) for r in groups[patient]]
        if len(indexes)!=len(set(indexes)): raise ValueError(f'Duplicate slice indices: {patient}')
    return dict(sorted(groups.items()))


def geometry_map(path):
    if not path:return {}
    obj=json.loads(Path(path).read_text())
    for pid,item in obj.items():
        if 'spacing_zyx' not in item:raise ValueError(f'Missing spacing_zyx for {pid}')
        if len(item['spacing_zyx'])!=3 or min(item['spacing_zyx'])<=0:raise ValueError(f'Invalid spacing for {pid}')
    return obj


def resized_binary_to_native(binary,shape):
    x=torch.from_numpy(binary.astype(np.float32))[None,None]
    return F.interpolate(x,size=shape,mode='nearest')[0,0].numpy().astype(bool)


def load_reference(records):
    """Native H5 mask stack. Returns (3,Z,H,W) booleans and indices."""
    indices=[];labels=[];original=None
    for row in records:
        _,mask=read_h5(row['path'])
        if original is None:original=mask.shape
        if mask.shape!=original:raise ValueError('Inconsistent in-plane geometry')
        indices.append(int(row['slice_index']))
        labels.append(np.stack([mask!=0,np.isin(mask,(1,4)),mask==4],axis=0))
    return np.stack(labels,axis=1),indices


def probabilities_for_patient(model,records,device,batch_size=16,size=160,workers=0):
    loader=DataLoader(H5Slices(records,size=size,augment=False),batch_size=batch_size,shuffle=False,num_workers=workers)
    arrays=[];model.eval()
    with torch.no_grad():
        for x,_ in loader:
            x=x.to(device)
            with torch.autocast('cuda',enabled=device.type=='cuda'):
                arrays.append(torch.sigmoid(model(x)).float().cpu().numpy())
    return np.concatenate(arrays,axis=0).transpose(1,0,2,3) # (3,Z,h,w)


def reconstruct(prob,shape):
    """Resize each channel of each slice back to original H,W, preserving probabilities."""
    x=torch.from_numpy(prob.astype(np.float32)).transpose(0,1) # Z,3,h,w
    return F.interpolate(x,size=shape,mode='bilinear',align_corners=False).transpose(0,1).numpy()


def require_full_sequence(records,expected_slices=None):
    idx=[int(r['slice_index']) for r in records]
    if idx!=list(range(idx[0],idx[0]+len(idx))):
        raise ValueError(f'Noncontiguous slices: {records[0]["patient_id"]}; cannot report volume metrics')
    if expected_slices is not None and len(idx)!=expected_slices:
        raise ValueError(f'Incomplete volume {records[0]["patient_id"]}: {len(idx)} != {expected_slices}')
    return idx
