import argparse
from pathlib import Path
import numpy as np
from src.experiment import read_manifest,load_checkpoint,get_device,save_csv,save_json
from src.volumes import group_patients,geometry_map,load_reference,probabilities_for_patient,reconstruct,require_full_sequence
from src.evaluation import postprocess,scores,aggregate

def run_evaluation(model,rows,out,settings,geometry=None,device=None,batch_size=16,size=160,workers=0,
                   expected_slices=155,require_spacing=True,save_predictions=False,compute_hd95=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    device=device or next(model.parameters()).device
    details=[];geom=geometry or {}
    for pid,records in group_patients(rows).items():
        require_full_sequence(records,expected_slices)
        true,idx=load_reference(records)
        spacing=geom.get(pid,{}).get('spacing_zyx')
        if require_spacing and compute_hd95 and spacing is None:
            raise ValueError(f'Physical spacing absent for patient {pid}; supply geometry JSON or --skip-hd95')
        probs=probabilities_for_patient(model,records,device,batch_size,size,workers)
        probs=reconstruct(probs,true.shape[-2:]);pred=postprocess(probs,settings['thresholds'],settings['min_component'])
        values=scores(pred,true,spacing or (1,1,1),compute_hd95=compute_hd95)
        for region,score in zip(('WT','TC','ET'),values):details.append(dict(patient_id=pid,region=region,**score))
        if save_predictions:np.savez_compressed(out/f'{pid}_prediction.npz',prediction=pred.astype(np.uint8),slice_index=idx)
        print(f'Evaluated {pid}',flush=True)
    save_csv(out/'per_patient.csv',details)
    summary=aggregate(details);summary['evaluation_settings']=settings
    summary['hd95_geometry']='spacing_zyx from provided original-image metadata' if compute_hd95 else 'not computed'
    save_json(out/'summary.json',summary)
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--split',default='test');p.add_argument('--calibration',required=True)
    p.add_argument('--geometry',help='JSON mapping patient_id to spacing_zyx in mm, e.g. {"patient":{"spacing_zyx":[1,1,1]}}')
    p.add_argument('--skip-hd95',action='store_true');p.add_argument('--allow-incomplete-volumes',action='store_true')
    p.add_argument('--out',default='runs/evaluation');p.add_argument('--save-predictions',action='store_true')
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--size',type=int,default=160)
    p.add_argument('--workers',type=int,default=0);p.add_argument('--device',default='auto');a=p.parse_args()
    settings=__import__('json').loads(Path(a.calibration).read_text());device=get_device(a.device)
    model,_=load_checkpoint(a.checkpoint,device)
    run_evaluation(model,read_manifest(a.manifest,a.split),a.out,settings,geometry_map(a.geometry),device,a.batch_size,a.size,a.workers,
        None if a.allow_incomplete_volumes else 155,not a.skip_hd95,a.save_predictions,not a.skip_hd95)
if __name__=='__main__':main()
