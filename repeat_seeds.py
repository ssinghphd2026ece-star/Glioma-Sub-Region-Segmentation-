"""Five independent training/calibration/test runs."""
import argparse,subprocess,sys,json
from pathlib import Path
import numpy as np
from src.experiment import save_csv,save_json

def cmd(args):subprocess.run([sys.executable,'-m',*args],check=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--config',default='configs/default.json')
    p.add_argument('--geometry');p.add_argument('--skip-hd95',action='store_true');p.add_argument('--out',default='runs/seeds');a=p.parse_args()
    records=[]
    for seed in (42,123,256,512,1024):
        d=Path(a.out)/str(seed);d.mkdir(parents=True,exist_ok=True)
        cmd(['scripts.train','--manifest',a.manifest,'--config',a.config,'--out',str(d),'--seed',str(seed)])
        cmd(['scripts.calibrate','--manifest',a.manifest,'--checkpoint',str(d/'best.pt'),'--out',str(d/'calibration.json')])
        args=['scripts.evaluate','--manifest',a.manifest,'--checkpoint',str(d/'best.pt'),'--calibration',str(d/'calibration.json'),'--out',str(d/'evaluation')]
        if a.geometry:args+=['--geometry',a.geometry]
        if a.skip_hd95:args+=['--skip-hd95']
        cmd(args)
        result=json.loads((d/'evaluation/summary.json').read_text())
        records.append(dict(seed=seed,**{k:v for k,v in result.items() if k.startswith('mean_')}))
        save_csv(Path(a.out)/'all_seeds.csv',records)
    metrics=[k for k in records[0] if k!='seed'];stat={}
    for k in metrics:
        v=[r[k] for r in records if r[k] is not None]
        stat[k]=dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1))) if len(v)>1 else None
    save_json(Path(a.out)/'seed_statistics.json',stat)
if __name__=='__main__':main()
