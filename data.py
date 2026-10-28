import re
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

def patient_id(path):
    # Default naming: BraTS20_Training_001_slice_042.h5; override via manifest if necessary.
    m=re.search(r'(BraTS\d+_(?:Training|Validation)_\d+)',Path(path).stem,re.I)
    if m:return m.group(1)
    m=re.search(r'(.+?)[_-](?:slice|z)[_-]?\d+$',Path(path).stem,re.I)
    if m:return m.group(1)
    raise ValueError(f'Cannot identify patient from {path}; provide a manifest with patient_id and slice_index')

def slice_index(path):
    m=re.search(r'(?:slice|z)[_-]?(\d+)$',Path(path).stem,re.I)
    if not m:raise ValueError(f'Cannot identify slice index: {path}')
    return int(m.group(1))

def read_h5(path):
    with h5py.File(path,'r') as f:
        if 'image' not in f or 'mask' not in f:raise KeyError(f'{path}: expected image and mask datasets')
        image=np.asarray(f['image'],dtype=np.float32);mask=np.asarray(f['mask'])
    if image.ndim!=3:raise ValueError(f'Expected HWC image: {path}: {image.shape}')
    if image.shape[-1]!=4 and image.shape[0]==4:image=np.moveaxis(image,0,-1)
    if image.shape[-1]!=4:raise ValueError(f'Expected four MRI modalities: {image.shape}')
    if mask.ndim==3 and mask.shape[-1]==1:mask=mask[...,0]
    if mask.ndim==3 and mask.shape[0]==1:mask=mask[0]
    if mask.ndim!=2:raise ValueError('Expected raw integer label mask, not pre-one-hot mask')
    labels=set(np.unique(mask).tolist())
    if not labels.issubset({0,1,2,4}):raise ValueError(f'Unexpected BraTS labels {labels}; conversion mapping must be documented')
    return image,mask

def transform(image,mask,size=160,augment=False):
    chans=[]
    for i in range(4):
        v=image[...,i].copy();fg=v!=0
        if fg.any():v[fg]=(v[fg]-v[fg].mean())/(v[fg].std()+1e-6)
        v[~fg]=0;chans.append(v)
    x=torch.from_numpy(np.stack(chans)).float()
    y=torch.from_numpy(np.stack([mask!=0,np.isin(mask,[1,4]),mask==4])).float()
    x=F.interpolate(x[None],size=(size,size),mode='bilinear',align_corners=False)[0]
    y=F.interpolate(y[None],size=(size,size),mode='nearest')[0]
    if augment:
        if torch.rand(())<.5:x=x.flip(-1);y=y.flip(-1)
        if torch.rand(())<.5:x=x.flip(-2);y=y.flip(-2)
        k=int(torch.randint(0,4,()).item());x=x.rot90(k,(-2,-1));y=y.rot90(k,(-2,-1))
        x=x*(1+0.1*torch.randn((4,1,1)))+0.05*torch.randn((4,1,1))+0.02*torch.randn_like(x)
    return x,y

class H5Slices(Dataset):
    def __init__(self,records,size=160,augment=False):self.records=records;self.size=size;self.augment=augment
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        r=self.records[i];image,mask=read_h5(r['path']);x,y=transform(image,mask,self.size,self.augment)
        return x,y
