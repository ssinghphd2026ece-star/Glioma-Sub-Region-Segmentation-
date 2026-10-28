import torch
from src.model import GliomaUNet
m=GliomaUNet().eval();print('Parameters:',sum(p.numel() for p in m.parameters()))
try:
 from thop import profile
 macs,params=profile(m,inputs=(torch.randn(1,4,160,160),),verbose=False)
 print('MACs:',macs,'FLOPs (2x MACs convention):',2*macs)
except ImportError:print('Install thop for approximate operation count')
