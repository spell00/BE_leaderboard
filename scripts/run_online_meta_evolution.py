#!/usr/bin/env python3
"""Online score-weighted neural meta-HPO seeded by a completed trial bank."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import wandb
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from scripts import hp_search
from src.zero_shot_recommender.meta_features import extract_meta_features, META_FEATURE_NAMES
DLOSS=("no","inverseTriplet","revTriplet","DANN","normae")
SCALER=("standard","robust","standard_per_batch","robust_per_batch")
NUM=("lr","wd","nu","smoothing","margin","dropout","thres","warmup","layer1","gamma","beta","class_triplet_w")
BOUNDS={"lr":(1e-4,1e-2,1),"wd":(1e-6,1e-3,1),"nu":(1e-4,1e2,1),"smoothing":(0,.2,0),"margin":(0,10,0),"dropout":(0,.5,0),"thres":(0,.1,0),"warmup":(1,50,0),"layer1":(512,1024,0),"gamma":(.01,100,1),"beta":(.01,100,1),"class_triplet_w":(1e-4,10,1)}
def args():
 p=argparse.ArgumentParser(); p.add_argument('--trial-bank',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--split-manifest',type=Path,default=ROOT/'config/evolution_development_datasets.json'); p.add_argument('--trials',type=int,default=100); p.add_argument('--patience',type=int,default=10); p.add_argument('--seed',type=int,default=42); p.add_argument('--hidden',type=int,default=64); p.add_argument('--meta-epochs',type=int,default=250); p.add_argument('--n-epochs',type=int,default=1000); p.add_argument('--n-repeats',type=int,default=3); p.add_argument('--num-workers',type=int,default=4); p.add_argument('--device',choices=('cpu','cuda'),default='cuda'); p.add_argument('--resume',action='store_true'); p.add_argument('--dry-run',action='store_true'); p.add_argument('--wandb-project',default='BE_leaderboard_meta_evolution'); p.add_argument('--wandb-run-name',default='online-meta-evolution-bank100'); return p.parse_args()
def norm(v,k):
 lo,hi,log=BOUNDS[k]; x=np.clip(float(v),lo,hi); return (np.log(x)-np.log(lo))/(np.log(hi)-np.log(lo)) if log else (x-lo)/(hi-lo or 1)
def denorm(x,k):
 lo,hi,log=BOUNDS[k]; x=float(np.clip(x,0,1)); return float(np.exp(np.log(lo)+x*(np.log(hi)-np.log(lo)))) if log else lo+x*(hi-lo)
class Net(nn.Module):
 def __init__(self,n,m): super().__init__(); self.body=nn.Sequential(nn.Linear(n,m),nn.ReLU(),nn.Linear(m,m),nn.ReLU()); self.num=nn.Linear(m,len(NUM)); self.dloss=nn.Linear(m,len(DLOSS)); self.scaler=nn.Linear(m,len(SCALER)); self.bits=nn.Linear(m,3); self.depth=nn.Linear(m,5)
 def forward(self,x):
  h=self.body(x); return self.num(h),self.dloss(h),self.scaler(h),self.bits(h),self.depth(h)
def load_bank(path): return [json.loads(x) for x in path.open() if x.strip() and json.loads(x).get('role')!='alzheimer_baseline' and json.loads(x).get('error') is None]
def main():
 a=args(); torch.manual_seed(a.seed); rng=np.random.default_rng(a.seed); a.output_dir.mkdir(parents=True,exist_ok=True); ledger=a.output_dir/'online_trials.jsonl'; statep=a.output_dir/'state.json'; wandb_run=wandb.init(project=a.wandb_project,name=a.wandb_run_name,config={'trials':a.trials,'patience':a.patience,'bank':str(a.trial_bank),'source':'online_score_weighted_neural_meta'},resume='allow',id='online-meta-evolution-bank100-v3')
 bank=load_bank(a.trial_bank); datasets=list(dict.fromkeys(r['dataset_id'] for r in bank)); meta=[]; loaded={}
 for d in datasets:
  X,y,b=hp_search.load_dataset(d); loaded[d]=(X,y,b); f=extract_meta_features(X,y,b); meta.append([f[n] for n in META_FEATURE_NAMES])
 meta=np.asarray(meta,dtype=np.float32); mu=np.nanmean(meta,0); sd=np.nanstd(meta,0); sd[sd<1e-8]=1; meta=(np.nan_to_num(meta,nan=0)-mu)/sd; dindex={d:i for i,d in enumerate(datasets)}
 records=[]
 if a.resume and ledger.exists(): records=[json.loads(x) for x in ledger.open() if x.strip()]
 best=max([float(r.get('aggregate_mcc',-1)) for r in records],default=-1); saved_state=json.loads(statep.read_text()) if a.resume and statep.exists() else {}; stale=int(saved_state.get('stale_trials',0)); start=int(saved_state.get('next_trial',20)) if a.resume else 20; hp=hp_search.parse_args([]); hp.n_epochs=a.n_epochs; hp.n_repeats=a.n_repeats; hp.num_workers=a.num_workers; hp.device=a.device
 for step in range(start,a.trials):
  online_rows=[{'dataset_id':d,'config':r['config'],'valid_mcc':v} for r in records for d,v in r.get('scores',{}).items()]
  rows=bank+online_rows; rows=[r for r in rows if r.get('config') and r.get('valid_mcc',r.get('aggregate_mcc')) is not None]
  X=torch.tensor(np.stack([meta[dindex[r['dataset_id']]] for r in rows]),dtype=torch.float32); y=np.asarray([float(r.get('valid_mcc',r.get('aggregate_mcc',-1))) for r in rows]); w=np.exp(np.clip((y-np.nanmax(y))/0.08,-8,0)).astype('float32'); w/=w.mean()
  targets=np.asarray([[norm(r['config'].get(k,0),k) for k in NUM] for r in rows],dtype=np.float32); net=Net(X.shape[1],a.hidden); opt=torch.optim.Adam(net.parameters(),lr=1e-2); net.train()
  for _ in range(a.meta_epochs):
   pn,pd,ps,pb,pe=net(X); loss=((pn-torch.tensor(targets))**2*torch.tensor(w[:,None])).mean(); opt.zero_grad(); loss.backward(); opt.step()
  target=torch.tensor(meta.mean(0)[None,:]); net.eval(); pn,pd,ps,pb,pe=net(target); cfg=dict(max(rows,key=lambda r:float(r.get('valid_mcc',r.get('aggregate_mcc',-1))))['config']); cfg.update({k:denorm(pn[0,i].item(),k) for i,k in enumerate(NUM)}); cfg['dloss']=DLOSS[int(pd[0].argmax())]; cfg['scaler']=SCALER[int(ps[0].argmax())]; cfg['n_layers']=int(pe[0].argmax())+1; cfg['warmup']=int(round(cfg['warmup'])); cfg['layer1']=int(round(cfg['layer1'])); cfg['variational']=bool(pb[0,0]>0); cfg['class_triplet']=bool(pb[0,1]>0); cfg['log1p']=bool(pb[0,2]>0); cfg['kan']=False
  # exploration/mutation prevents deterministic collapse
  if step and rng.random()<.35:
   k=rng.choice(NUM); lo,hi,_=BOUNDS[k]; cfg[k]=denorm(np.clip(norm(cfg[k],k)+rng.normal(0,.12),0,1),k)
  scores={}; errors={}
  for d in datasets:
   if a.dry_run: scores[d]=float(np.tanh(np.mean(meta[dindex[d]])*.01+rng.normal(0,.01))); continue
   Xd,yd,bd=loaded[d]; meta_run=argparse.Namespace(**vars(hp)); meta_run.dataset=d; meta_run.seed=a.seed+step; meta_run.results_dir=str(a.output_dir/d); meta_run.cv_split_cache=str(a.output_dir/'cv_splits'/f'{d}.npz'); meta_run.resolved_n_repeats=hp_search.resolve_n_repeats(meta_run.n_repeats,bd)
   try: scores[d],_=hp_search.run_trial(cfg,meta_run,(Xd,yd,bd),f'online_meta_evolution_{step}_{d}')
   except Exception as e: scores[d]=-1.; errors[d]=f'{type(e).__name__}: {e}'
  agg=float(np.mean(list(scores.values()))); rec={'trial':step,'config':cfg,'scores':scores,'aggregate_mcc':agg,'best_before':best,'stale_before':stale,'errors':errors,'source':'online_score_weighted_neural_meta'}
  with ledger.open('a') as fh: fh.write(json.dumps(rec,default=str)+'\n'); fh.flush(); os.fsync(fh.fileno())
  if agg>best: best=agg; stale=0
  else: stale+=1
  state={'next_trial':step+1,'best_aggregate_mcc':best,'stale_trials':stale,'max_trials':a.trials,'patience':a.patience,'datasets':datasets,'seed':a.seed,'bank':str(a.trial_bank)}; statep.write_text(json.dumps(state,indent=2)+'\n')
  print(f'[online-meta] trial={step+1}/{a.trials} aggregate={agg:.4f} best={best:.4f} stale={stale}/{a.patience}',flush=True)
  wandb_run.log({"trial":step,"aggregate_mcc":agg,"best_aggregate_mcc":best,"stale_trials":stale,"config":cfg,**{f"score/{k}":v for k,v in scores.items()}})
  if stale>=a.patience: break
 wandb_run.finish()
 return 0
if __name__=='__main__': raise SystemExit(main())
