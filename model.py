import torch
from torch import nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(ci,co,3,padding=1,bias=False),nn.BatchNorm2d(co),nn.ReLU(inplace=True),nn.Conv2d(co,co,3,padding=1,bias=False),nn.BatchNorm2d(co))
        self.skip=nn.Identity() if ci==co else nn.Conv2d(ci,co,1,bias=False)
        self.act=nn.ReLU(inplace=True)
    def forward(self,x): return self.act(self.net(x)+self.skip(x))

class CBAM(nn.Module):
    def __init__(self,c,r=16):
        super().__init__()
        self.mlp=nn.Sequential(nn.Conv2d(c,max(1,c//r),1),nn.ReLU(inplace=True),nn.Conv2d(max(1,c//r),c,1))
        self.spatial=nn.Conv2d(2,1,7,padding=3)
    def forward(self,x):
        x=x*torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x,1))+self.mlp(F.adaptive_max_pool2d(x,1)))
        return x*torch.sigmoid(self.spatial(torch.cat([x.mean(1,keepdim=True),x.amax(1,keepdim=True)],1)))

class Gate(nn.Module):
    def __init__(self,cs,cg):
        super().__init__(); mid=max(1,cs//2)
        self.s=nn.Conv2d(cs,mid,1); self.g=nn.Conv2d(cg,mid,1); self.psi=nn.Conv2d(mid,1,1)
    def forward(self,s,g): return s*torch.sigmoid(self.psi(F.relu(self.s(s)+self.g(g))))

class ASPP(nn.Module):
    def __init__(self,c):
        super().__init__(); b=max(16,c//4)
        self.branches=nn.ModuleList([nn.Sequential(nn.Conv2d(c,b,1 if d==1 else 3,padding=0 if d==1 else d,dilation=d,bias=False),nn.BatchNorm2d(b),nn.ReLU(inplace=True)) for d in (1,2,4,8)])
        self.pool=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(c,b,1),nn.ReLU(inplace=True))
        self.fuse=nn.Conv2d(5*b,c,1)
    def forward(self,x):
        p=F.interpolate(self.pool(x),size=x.shape[-2:],mode='bilinear',align_corners=False)
        return F.relu(x+self.fuse(torch.cat([*(branch(x) for branch in self.branches),p],1)))

class GliomaUNet(nn.Module):
    def __init__(self,channels=(32,64,128,256,512),aspp=True,cbam=True,gates=True):
        super().__init__(); self.use_gates=gates
        self.enc=nn.ModuleList(); self.enc_att=nn.ModuleList(); prev=4
        for c in channels:
            self.enc.append(Residual(prev,c));self.enc_att.append(CBAM(c) if cbam else nn.Identity());prev=c
        self.pool=nn.MaxPool2d(2)
        self.bottleneck=ASPP(channels[-1]) if aspp else nn.Identity()
        self.up=nn.ModuleList();self.gates=nn.ModuleList();self.dec=nn.ModuleList();self.dec_att=nn.ModuleList()
        for c in reversed(channels[:-1]):
            self.up.append(nn.ConvTranspose2d(prev,c,2,stride=2));self.gates.append(Gate(c,c) if gates else nn.Identity())
            self.dec.append(Residual(c*2,c));self.dec_att.append(CBAM(c) if cbam else nn.Identity());prev=c
        self.out=nn.Conv2d(prev,3,1)
    def forward(self,x):
        skips=[]
        for i,(block,att) in enumerate(zip(self.enc,self.enc_att)):
            x=att(block(x));skips.append(x)
            if i<len(self.enc)-1:x=self.pool(x)
        x=self.bottleneck(x)
        for up,gate,dec,att,s in zip(self.up,self.gates,self.dec,self.dec_att,reversed(skips[:-1])):
            x=up(x)
            if x.shape[-2:]!=s.shape[-2:]:x=F.interpolate(x,size=s.shape[-2:],mode='bilinear',align_corners=False)
            s=gate(s,x) if self.use_gates else s
            x=att(dec(torch.cat([x,s],dim=1)))
        return self.out(x)
