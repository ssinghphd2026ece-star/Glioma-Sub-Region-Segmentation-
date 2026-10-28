import argparse,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--config',default='configs/default.json');a=p.parse_args()
for name,asp,cb,ag in [('unet',0,0,0),('aspp',1,0,0),('cbam',0,1,0),('ag',0,0,1),('aspp_cbam',1,1,0),('aspp_ag',1,0,1),('cbam_ag',0,1,1),('full',1,1,1)]:
 subprocess.run([sys.executable,'-m','scripts.train','--manifest',a.manifest,'--config',a.config,'--out',f'runs/{name}','--aspp',str(asp),'--cbam',str(cb),'--gates',str(ag)],check=True)
