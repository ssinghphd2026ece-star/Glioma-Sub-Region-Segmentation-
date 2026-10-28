"""Generate publication-friendly plots strictly from actual experiment outputs."""
import argparse,csv,json
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs');p.add_argument('--out',default='figures');a=p.parse_args()
    root=Path(a.runs);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for path in sorted(root.glob('*/evaluation/summary.json')):
        s=json.loads(path.read_text());summaries.append((path.parent.parent.name,s))
    for path in sorted(root.glob('*/history.csv')):
        with open(path) as f:rows=list(csv.DictReader(f))
        if not rows:continue
        x=[int(r['epoch']) for r in rows]
        fig,ax=plt.subplots(figsize=(7,4));ax.plot(x,[float(r['train_loss']) for r in rows],label='Training loss')
        ax.plot(x,[float(r['val_loss']) for r in rows],label='Validation loss')
        ax.set(xlabel='Epoch',ylabel='Loss',title=path.parent.name+' training');ax.legend();fig.tight_layout()
        fig.savefig(out/f'{path.parent.name}_loss.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4));ax.plot(x,[float(r['val_dice']) for r in rows],label='Validation soft-selection Dice at 0.5')
        ax.set(xlabel='Epoch',ylabel='Dice',title=path.parent.name+' validation');ax.legend();fig.tight_layout()
        fig.savefig(out/f'{path.parent.name}_dice.png',dpi=180);plt.close(fig)
    if summaries:
        names=[n for n,_ in summaries];v=[s.get('mean_dice',np.nan) for _,s in summaries]
        fig,ax=plt.subplots(figsize=(max(8,len(names)*.9),4));ax.bar(names,v);ax.set(ylabel='Patient-level mean Dice',ylim=(0,1));ax.tick_params(axis='x',rotation=40);fig.tight_layout()
        fig.savefig(out/'ablation_mean_dice.png',dpi=180);plt.close(fig)
    print('Saved figures in',out,'; all figures based on measured outputs only')
if __name__=='__main__':main()
