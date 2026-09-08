#!/usr/bin/env python3
"""Twenty bank-index meta trials, then adaptive Alzheimer-only continuation."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(ROOT))
from scripts import hp_search
from src.meta_hpo_bank import load_bank,source_dataset_ids
from src.meta_hpo_models import train_direct_meta_model,normalize_source_meta,_decode_direct
from src.zero_shot_recommender.meta_features import extract_meta_features,META_FEATURE_NAMES
def perturb(c,rng):
 c=dict(c)
 for k,scale in (("lr",.25),("wd",.35),("nu",.35),("smoothing",.12),("margin",.18),("dropout",.18),("thres",.25),("gamma",.35),("beta",.35),("class_triplet_w",.35)):
  if k in c and rng.random()<.7:
   if k in ("lr","wd","nu","gamma","beta","class_triplet_w"): c[k]=float(np.exp(np.log(max(float(c[k]),1e-8))+rng.normal(0,scale)))
   else: c[k]=float(c[k])+float(rng.normal(0,scale))
 c["lr"]=float(np.clip(c.get("lr",1e-3),1e-4,1e-2)); c["wd"]=float(np.clip(c.get("wd",1e-5),1e-6,1e-3)); c["nu"]=float(np.clip(c.get("nu",1),1e-4,100)); c["dropout"]=float(np.clip(c.get("dropout",.2),0,.5)); c["smoothing"]=float(np.clip(c.get("smoothing",.05),0,.2)); c["margin"]=float(np.clip(c.get("margin",1),0,10)); c["thres"]=float(np.clip(c.get("thres",0),0,.1)); c["warmup"]=int(np.clip(round(c.get("warmup",20)),1,50)); c["layer1"]=int(np.clip(round(c.get("layer1",768)),512,1024)); return c
def main():
 p=argparse.ArgumentParser(); p.add_argument("--trial-bank",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--trials",type=int,default=100); p.add_argument("--patience",type=int,default=10); p.add_argument("--meta-epochs",type=int,default=1000); p.add_argument("--n-epochs",type=int,default=1000); p.add_argument("--n-repeats",type=int,default=3); p.add_argument("--hidden-size",type=int,default=64); p.add_argument("--seed",type=int,default=42); p.add_argument("--num-workers",type=int,default=4); p.add_argument("--device",default="cuda"); p.add_argument("--wandb-project",default="BE_leaderboard_meta_evolution"); p.add_argument("--wandb-run-name",default="meta-train3-bank20-online-alzheimer"); a=p.parse_args()
 a.output_dir.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(a.seed)
 trials=load_bank(a.trial_bank); sources=source_dataset_ids(trials); valid="massbench_benchmark"; alz="massbench_alzheimer"; train_ids=tuple(x for x in sources if x!=valid)
 data={x:hp_search.load_dataset(x) for x in sources+(alz,)}; meta={}
 for x,(X,y,b) in data.items():
  f=extract_meta_features(X,y,b); meta[x]=np.asarray([f[n] for n in META_FEATURE_NAMES],dtype=np.float32)
 norm,mean,scale=normalize_source_meta(meta,train_ids); max_warmup=max(1,min(50,a.n_epochs))
 by={d:sorted([r for r in trials if r.dataset_id==d and r.role=="source"],key=lambda r:r.trial_index) for d in sources}
 hp=hp_search.parse_args([]); hp.n_epochs=a.n_epochs; hp.n_repeats=a.n_repeats; hp.num_workers=a.num_workers; hp.device=a.device
 import wandb; wr=wandb.init(project=a.wandb_project,name=a.wandb_run_name,config={"protocol":"20_bank_index_meta_trials_then_adaptive_alzheimer","train_datasets":train_ids,"benchmark":valid,"target":alz,"trials":a.trials,"patience":a.patience,"bank":str(a.trial_bank)})
 ledger=a.output_dir/"trials.jsonl"; best=-np.inf; stale=0; best_cfg=None
 def eval_alz(cfg,step):
  X,y,b=data[alz]; run=argparse.Namespace(**vars(hp)); run.dataset=alz; run.seed=a.seed+step; run.results_dir=str(a.output_dir/"alzheimer"); run.cv_split_cache=str(a.output_dir/"cv_splits"/"massbench_alzheimer.npz"); run.resolved_n_repeats=hp_search.resolve_n_repeats(run.n_repeats,b)
  try: score,metrics=hp_search.run_trial(cfg,run,(X,y,b),f"meta_train3_bank20_step{step}_alzheimer"); return float(score),metrics,None
  except Exception as e: return -1.,{},f"{type(e).__name__}: {e}"
 for step in range(a.trials):
  if step<20:
   chosen={d:by[d][step] for d in train_ids}; model,hist,diag=train_direct_meta_model(chosen,norm,meta[alz],max_warmup=max_warmup,hidden_size=a.hidden_size,epochs=a.meta_epochs,lr=1e-2,seed=a.seed+step)
   cfg=hist[-1].target_config; model.eval()
   with torch.no_grad(): pred=_decode_direct(model.forward(torch.tensor(((meta[valid]-mean)/scale)[None,:],dtype=torch.float32)),0,max_warmup,{})
   ref=by[valid][step].config; err=float(np.mean([float(pred.get(k)!=ref.get(k)) if not isinstance(pred.get(k,0),(int,float)) else abs(float(pred.get(k,0))-float(ref.get(k,0))) for k in ref]))
   phase="bank_index"; train_loss=hist[-1].train_loss
  else:
   if best_cfg is None: break
   cfg=perturb(best_cfg,rng); pred={}; err=None; phase="adaptive"; train_loss=None
  score,metrics,error=eval_alz(cfg,step); improved=score>best
  if improved: best=score; stale=0; best_cfg=dict(cfg)
  else: stale+=1
  rec={"trial":step+1,"phase":phase,"bank_index":step if step<20 else None,"train_datasets":train_ids,"benchmark_predicted_config":pred,"benchmark_config_error":err,"meta_train_loss":train_loss,"alzheimer_config":cfg,"alzheimer_valid_mcc":score,"alzheimer_metrics":metrics,"error":error,"best_alzheimer_valid_mcc":best,"stale_trials":stale}
  with ledger.open("a") as f: f.write(json.dumps(rec,default=str)+chr(10)); f.flush(); os.fsync(f.fileno())
  state={"next_trial":step+1,"best_alzheimer_valid_mcc":best,"stale_trials":stale,"max_trials":a.trials,"patience":a.patience,"bank_indices_completed":min(step+1,20),"train_datasets":train_ids,"benchmark":valid,"target":alz}; (a.output_dir/"state.json").write_text(json.dumps(state,indent=2)+chr(10))
  payload={"trial":step+1,"alzheimer_valid_mcc":score,"best_alzheimer_valid_mcc":best,"stale_trials":stale}
  if err is not None: payload["benchmark_config_error"]=err
  wr.log(payload); print(f"[meta-train3] trial={step+1}/{a.trials} phase={phase} Alzheimer={score:.4f} best={best:.4f} stale={stale}/{a.patience}",flush=True)
  if stale>=a.patience: break
 wr.finish()
if __name__=="__main__": main()
