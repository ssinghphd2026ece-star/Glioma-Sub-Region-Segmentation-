"""Export one patient's WT/TC/ET masks and optional overlay of one axial slice."""
import argparse,json
from pathlib import Path
import numpy as np
from src.experiment import read_manifest,load_checkpoint,get_device
from src.volumes import group_patients,load_reference,probabilities_for_patient,reconstruct
from src.evaluation import postprocess
from src.data import read_h5

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--patient',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--calibration',required=True)
    p.add_argument('--out',default='prediction.npz');p.add_argument('--device',default='auto');p.add_argument('--slice',type=int)
    a=p.parse_args();rows=group_patients(read_manifest(a.manifest))
    if a.patient not in rows:raise ValueError('Unknown patient ID')
    model,_=load_checkpoint(a.checkpoint,get_device(a.device));reference,idx=load_reference(rows[a.patient])
    probs=reconstruct(probabilities_for_patient(model,rows[a.patient],next(model.parameters()).device),reference.shape[-2:])
    cfg=json.loads(Path(a.calibration).read_text());pred=postprocess(probs,cfg['thresholds'],cfg['min_component'])
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,prediction=pred.astype('uint8'),reference=reference.astype('uint8'),slice_indices=idx)
    if a.slice is not None:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        if a.slice not in idx:raise ValueError('Slice index not present')
        k=idx.index(a.slice);im,_=read_h5(rows[a.patient][k]['path'])
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        axes[0].imshow(im[...,0],cmap='gray');axes[0].set_title('MRI channel 0')
        axes[1].imshow(reference[0,k],cmap='gray');axes[1].set_title('WT ground truth')
        axes[2].imshow(pred[0,k],cmap='gray');axes[2].set_title('WT predicted')
        for ax in axes:ax.axis('off')
        fig.tight_layout();fig.savefig(str(Path(a.out).with_suffix('.png')),dpi=180);plt.close(fig)
    print('Saved',a.out)
if __name__=='__main__':main()
