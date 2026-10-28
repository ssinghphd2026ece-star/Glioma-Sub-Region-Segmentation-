import argparse,csv,json,random
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from src.model import GliomaUNet
from src.data import H5Slices
from src.losses import composite_loss
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--config',default='configs/default.json');p.add_argument('--out',default='runs/full');p.add_argument('--seed',type=int,default=42);p.add_argument('--aspp',type=int,default=1);p.add_argument('--cbam',type=int,default=1);p.add_argument('--gates',type=int,default=1);a=p.parse_args()
cfg=json.load(open(a.config));random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
rows=list(csv.DictReader(open(a.manifest)));out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
json.dump(dict(config=cfg,arguments=vars(a)),open(out/'run_config.json','w'),indent=2)
device='cuda' if torch.cuda.is_available() else 'cpu';model=GliomaUNet(aspp=bool(a.aspp),cbam=bool(a.cbam),gates=bool(a.gates)).to(device)
train=DataLoader(H5Slices([r for r in rows if r['split']=='train'],size=cfg['image_size'],augment=True),batch_size=cfg['batch_size'],shuffle=True,num_workers=cfg['workers'])
val=DataLoader(H5Slices([r for r in rows if r['split']=='val'],size=cfg['image_size']),batch_size=cfg['batch_size'],num_workers=cfg['workers'])
opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=cfg['scheduler_patience'])
scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda');best=-1;wait=0;history=[]
for epoch in range(cfg['epochs']):
 model.train();tl=[]
 for x,y in train:
  x=x.to(device);y=y.to(device);opt.zero_grad(set_to_none=True)
  with torch.autocast(device_type=device,enabled=device=='cuda'):
   logits=model(x);loss=composite_loss(logits,y,**cfg['loss'])
  scaler.scale(loss).backward();scaler.step(opt);scaler.update();tl.append(loss.item())
 model.eval();vl=[];inter=np.zeros(3);den=np.zeros(3)
 with torch.no_grad():
  for x,y in val:
   x=x.to(device);y=y.to(device);logits=model(x);vl.append(composite_loss(logits,y,**cfg['loss']).item());pred=(logits.sigmoid()>.5)
   inter+=(2*(pred*y).sum((0,2,3))).cpu().numpy();den+=(pred.sum((0,2,3))+y.sum((0,2,3))).cpu().numpy()
 score=float(np.mean((inter+1e-6)/(den+1e-6)));v=float(np.mean(vl));sched.step(v)
 history.append(dict(epoch=epoch+1,train_loss=float(np.mean(tl)),val_loss=v,val_dice=score));print(history[-1],flush=True)
 if score>best:best=score;wait=0;torch.save({'model':model.state_dict(),'aspp':bool(a.aspp),'cbam':bool(a.cbam),'gates':bool(a.gates),'seed':a.seed},out/'best.pt')
 else:wait+=1
 with open(out/'history.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=history[0].keys());w.writeheader();w.writerows(history)
 if wait>=cfg['early_stop_patience']:break
