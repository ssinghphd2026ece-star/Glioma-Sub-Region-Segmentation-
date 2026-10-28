import argparse,csv
from pathlib import Path
import h5py,numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='demo_data');p.add_argument('--slices',type=int,default=3)
    a=p.parse_args();root=Path(a.out);root.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(42)
    with open(root/'manifest.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','patient_id','slice_index','split','has_tumor']);w.writeheader()
        for split,pid in [('train','Demo_001'),('val','Demo_002'),('test','Demo_003')]:
            for z in range(a.slices):
                img=rng.normal(size=(32,32,4)).astype('float32');img[:3]=0
                lab=np.zeros((32,32),dtype='uint8');lab[8:24,8:24]=2;lab[11:21,11:21]=1;lab[14:18,14:18]=4
                path=root/f'{pid}_slice_{z:03d}.h5'
                with h5py.File(path,'w') as h:h['image']=img;h['mask']=lab
                w.writerow(dict(path=path.resolve(),patient_id=pid,slice_index=z,split=split,has_tumor=1))
    print('Synthetic smoke-test manifest:',root/'manifest.csv')
if __name__=='__main__':main()
