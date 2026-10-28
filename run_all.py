"""End-to-end eight configurations, training -> calibration -> evaluation."""
import argparse,subprocess,sys
from pathlib import Path

def execute(*parts):subprocess.run([sys.executable,'-m',*parts],check=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--config',default='configs/default.json')
    p.add_argument('--geometry');p.add_argument('--skip-hd95',action='store_true');p.add_argument('--out',default='runs/ablation');p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    for name,asp,cb,ag in [('unet',0,0,0),('aspp',1,0,0),('cbam',0,1,0),('ag',0,0,1),('aspp_cbam',1,1,0),('aspp_ag',1,0,1),('cbam_ag',0,1,1),('full',1,1,1)]:
        d=Path(a.out)/name;d.mkdir(parents=True,exist_ok=True)
        execute('scripts.train','--manifest',a.manifest,'--config',a.config,'--out',str(d),'--seed',str(a.seed),'--aspp',str(asp),'--cbam',str(cb),'--gates',str(ag))
        execute('scripts.calibrate','--manifest',a.manifest,'--checkpoint',str(d/'best.pt'),'--out',str(d/'calibration.json'))
        args=['scripts.evaluate','--manifest',a.manifest,'--checkpoint',str(d/'best.pt'),'--calibration',str(d/'calibration.json'),'--out',str(d/'evaluation')]
        if a.geometry:args+=['--geometry',a.geometry]
        if a.skip_hd95:args+=['--skip-hd95']
        execute(*args)
if __name__=='__main__':main()
