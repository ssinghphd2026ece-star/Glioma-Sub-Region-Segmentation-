import torch
import torch.nn.functional as F

def composite_loss(logits,target,alpha=.7,beta=.3,gamma=1.33,weights=(1.,1.,1.),pos_weight=3.,eps=1e-6):
    p=torch.sigmoid(logits);dims=(0,2,3)
    tp=(p*target).sum(dims);fp=(p*(1-target)).sum(dims);fn=((1-p)*target).sum(dims)
    dice=1-(2*tp+eps)/(2*tp+fp+fn+eps)
    tv=(tp+eps)/(tp+alpha*fn+beta*fp+eps)
    focal=(1-tv).pow(gamma)
    wbce=F.binary_cross_entropy_with_logits(logits,target,pos_weight=torch.tensor(pos_weight,device=logits.device))
    a,b,c=weights
    return a*focal.mean()+b*dice.mean()+c*wbce
