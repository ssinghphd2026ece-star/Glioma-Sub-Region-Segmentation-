import torch
from src.model import GliomaUNet
from src.losses import composite_loss
from src.evaluation import postprocess
import numpy as np

def test_forward_and_backward():
 m=GliomaUNet(channels=(8,16,32,64,128));x=torch.randn(1,4,32,32);y=torch.zeros(1,3,32,32)
 o=m(x);assert o.shape==y.shape;composite_loss(o,y).backward()

def test_hierarchy():
 p=np.zeros((3,2,4,4));p[2,0,1,1]=1
 m=postprocess(p);assert m[0,0,1,1] and m[1,0,1,1]
