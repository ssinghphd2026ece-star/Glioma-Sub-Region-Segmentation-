import numpy as np
from src.evaluation import postprocess,scores,aggregate
from src.volumes import reconstruct,require_full_sequence
from src.losses import composite_loss
import torch

def test_nested_regions_and_scoring():
    prob=np.zeros((3,2,8,8),dtype=np.float32)
    prob[2,:,2:5,2:5]=.95
    mask=postprocess(prob,(.5,.5,.5),(0,0,0))
    assert np.array_equal(mask[0],mask[1])
    assert np.array_equal(mask[1],mask[2])
    measured=scores(mask,mask,compute_hd95=False)
    assert all(x['dice']==1 for x in measured)
    assert aggregate([dict(patient_id='p',region=k,**v) for k,v in zip(('WT','TC','ET'),measured)])['mean_dice']==1.

def test_resize_and_contiguity():
    x=np.random.default_rng(1).random((3,3,8,8),dtype=np.float32)
    assert reconstruct(x,(24,24)).shape==(3,3,24,24)
    rows=[{'slice_index':str(i),'patient_id':'p'} for i in range(3)]
    require_full_sequence(rows,3)

def test_loss_backprop():
    x=torch.randn(2,3,16,16,requires_grad=True)
    y=torch.zeros_like(x);y[:,:,3:9,3:9]=1
    v=composite_loss(x,y);v.backward()
    assert torch.isfinite(v) and torch.isfinite(x.grad).all()
