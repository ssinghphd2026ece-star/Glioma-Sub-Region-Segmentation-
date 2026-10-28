"""3D patient-level segmentation metrics, surface HD95, and post-processing."""
import numpy as np
from scipy.ndimage import label,binary_erosion,distance_transform_edt

def postprocess(prob,thresholds=(.5,.5,.5),min_component=(0,0,0)):
    """prob: (3,Z,H,W); filter each region then enforce ET subset TC subset WT."""
    thresholds=np.asarray(thresholds,dtype=float)
    if thresholds.shape!=(3,):raise ValueError('Expected 3 region thresholds')
    cuts=[int(min_component)]*3 if np.isscalar(min_component) else list(min_component)
    if len(cuts)!=3:raise ValueError('Expected 3 component size rules')
    masks=prob>=thresholds[:,None,None,None]
    for c in range(3):
        cc,n=label(masks[c],structure=np.ones((3,3,3),dtype=np.uint8))
        if cuts[c]>0:
            sizes=np.bincount(cc.ravel());masks[c][(cc>0)&(sizes[cc]<cuts[c])]=False
    masks[1]|=masks[2];masks[0]|=masks[1]
    return masks

def _hd95(p,t,spacing):
    if not p.any() and not t.any():return 0.
    if not p.any() or not t.any():return float('nan')
    structure=np.ones((3,3,3),dtype=bool)
    ps=p & ~binary_erosion(p,structure=structure,border_value=0)
    ts=t & ~binary_erosion(t,structure=structure,border_value=0)
    dt=distance_transform_edt(~ts,sampling=spacing)
    dp=distance_transform_edt(~ps,sampling=spacing)
    return float(max(np.percentile(dt[ps],95),np.percentile(dp[ts],95)))

def scores(pred,truth,spacing=(1,1,1),compute_hd95=True):
    """Both (3,Z,H,W). HD95=NaN for exactly one empty volume."""
    if pred.shape!=truth.shape or pred.ndim!=4 or pred.shape[0]!=3:raise ValueError('Expected matching (3,Z,H,W)')
    out=[]
    for p,t in zip(pred.astype(bool),truth.astype(bool)):
        tp=int(np.logical_and(p,t).sum());fp=int(np.logical_and(p,~t).sum())
        fn=int(np.logical_and(~p,t).sum());tn=int(np.logical_and(~p,~t).sum())
        den=2*tp+fp+fn;union=tp+fp+fn
        out.append(dict(dice=2*tp/den if den else 1.,iou=tp/union if union else 1.,
            precision=tp/(tp+fp) if tp+fp else (1. if not t.any() else 0.),
            sensitivity=tp/(tp+fn) if tp+fn else (1. if not p.any() else 0.),
            specificity=tn/(tn+fp) if tn+fp else 1.,
            accuracy=(tp+tn)/(tp+tn+fp+fn),
            hd95_mm=_hd95(p,t,spacing) if compute_hd95 else None))
    return out

def aggregate(patient_rows):
    if not patient_rows:return {}
    regions=('WT','TC','ET');result={'patients':len(set(r['patient_id'] for r in patient_rows))}
    for region in regions:
        subset=[r for r in patient_rows if r['region']==region]
        for metric in ('dice','iou','precision','sensitivity','specificity','accuracy','hd95_mm'):
            values=[float(r[metric]) for r in subset if r.get(metric) is not None and np.isfinite(float(r[metric]))]
            result[f'{region}_{metric}']=float(np.mean(values)) if values else None
            if metric=='hd95_mm':result[f'{region}_hd95_valid_patients']=len(values)
    for metric in ('dice','iou','precision','sensitivity','specificity','accuracy','hd95_mm'):
        values=[result[f'{r}_{metric}'] for r in regions if result[f'{r}_{metric}'] is not None]
        result[f'mean_{metric}']=float(np.mean(values)) if values else None
    return result
