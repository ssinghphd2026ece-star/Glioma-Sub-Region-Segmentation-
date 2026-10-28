import argparse
from pathlib import Path
import numpy as np
from src.experiment import read_manifest,load_checkpoint,get_device,save_json
from src.volumes import group_patients,load_reference,probabilities_for_patient,reconstruct,require_full_sequence
from src.evaluation import postprocess,scores

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--out',default='runs/full/calibration.json');p.add_argument('--thresholds',nargs='+',type=float,default=[.3,.4,.5,.6,.7])
    p.add_argument('--component-sizes',nargs='+',type=int,default=[0,10,50,100])
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--size',type=int,default=160)
    p.add_argument('--device',default='auto');p.add_argument('--allow-incomplete-volumes',action='store_true');a=p.parse_args()
    model,_=load_checkpoint(a.checkpoint,get_device(a.device))
    cached=[]
    for pid,rows in group_patients(read_manifest(a.manifest,'val')).items():
        require_full_sequence(rows,None if a.allow_incomplete_volumes else 155)
        truth,_=load_reference(rows)
        pmap=reconstruct(probabilities_for_patient(model,rows,next(model.parameters()).device,a.batch_size,a.size),truth.shape[-2:])
        cached.append((pmap,truth));print('Cached validation patient',pid,flush=True)
    if not cached:raise ValueError('No validation patients')
    # Coordinate descent over jointly evaluated postprocessed Dice; no test data used.
    threshold=[.5]*3;minimum=[0]*3
    for iteration in range(2):
        for region in range(3):
            best=(-1.,None,None)
            for t in a.thresholds:
                for cut in a.component_sizes:
                    th=threshold.copy();mi=minimum.copy();th[region]=t;mi[region]=cut
                    dices=[scores(postprocess(p,th,mi),truth,compute_hd95=False)[region]['dice'] for p,truth in cached]
                    score=float(np.mean(dices))
                    if score>best[0]:best=(score,t,cut)
            threshold[region]=best[1];minimum[region]=best[2]
            print(f'Iteration {iteration+1} region {region}: {best}',flush=True)
    save_json(a.out,dict(thresholds=threshold,min_component=minimum,
        tuning='validation-only coordinate search; 3D connected components; voxel sizes',
        validation_patients=len(cached),threshold_grid=a.thresholds,component_grid=a.component_sizes))
if __name__=='__main__':main()
