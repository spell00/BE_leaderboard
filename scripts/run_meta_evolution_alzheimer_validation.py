#!/usr/bin/env python3
"""Evolutionary meta-network arm: three-source fit, benchmark diagnostic, Alzheimer validation."""
import argparse,json,os,sys,time,copy
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(ROOT))
from scripts import hp_search
from src.meta_hpo_bank import load_bank,source_dataset_ids,best_by_dataset,trials_at_prefix
from src.meta_hpo_models import train_direct_meta_model,normalize_source_meta,_decode_direct
from src.zero_shot_recommender.meta_features import extract_meta_features,META_FEATURE_NAMES
from src.meta_leaderboard import update_best
def main():
 p=argparse.ArgumentParser(); p.add_argument("--trial-bank",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--trials",type=int,default=100); p.add_argument("--patience",type=int,default=10); p.add_argument("--meta-epochs",type=int,default=300); p.add_argument("--hidden-size",type=int,default=64); p.add_argument("--n-epochs",type=int,default=1000); p.add_argument("--n-repeats",type=int,default=3); p.add_argument("--num-workers",type=int,default=4); p.add_argument("--device",default="cuda"); p.add_argument("--seed",type=int,default=42); p.add_argument("--resume",action="store_true"); p.add_argument("--wandb-project",default="BE_leaderboard_meta_evolution"); p.add_argument("--wandb-run-name",default="meta-evolution-v1-alzheimer-validation")
 a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True); torch.manual_seed(a.seed); rng=np.random.default_rng(a.seed)
 bank=load_bank(a.trial_bank); sources=source_dataset_ids(bank); valid="massbench_benchmark"; alz="massbench_alzheimer"; train_ids=tuple(x for x in sources if x!=valid)
 data={x:hp_search.load_dataset(x) for x in sources+(alz,)}; meta={}
 for x,(X,y,b) in data.items():
  f=extract_meta_features(X,y,b); meta[x]=np.asarray([f[n] for n in META_FEATURE_NAMES],dtype=np.float32)
 norm,mean,scale=normalize_source_meta(meta,train_ids); max_warmup=max(1,min(50,a.n_epochs)); bests=best_by_dataset(trials_at_prefix(bank,20,role="source"),sources)
 bt={d:bests[d] for d in train_ids}; base,hist,diag=train_direct_meta_model(bt,norm,meta[alz],max_warmup=max_warmup,hidden_size=a.hidden_size,epochs=a.meta_epochs,lr=1e-2,seed=a.seed)
 zvalid=torch.tensor(((meta[valid]-mean)/scale)[None,:],dtype=torch.float32); zalz=torch.tensor(((meta[alz]-mean)/scale)[None,:],dtype=torch.float32)
 ledger=a.output_dir/"evolution_trials.jsonl"; statep=a.output_dir/"state.json"; modelp=a.output_dir/"best_model.pt"; rows=[json.loads(x) for x in ledger.open()] if a.resume and ledger.exists() else []
 start=len(rows); best_score=max([float(r.get("alzheimer_valid_mcc",-1)) for r in rows],default=-1); stale=0 if not rows else int(json.loads(statep.read_text()).get("stale_trials",0)); best_state=torch.load(modelp,map_location="cpu") if a.resume and modelp.exists() else copy.deepcopy(base.state_dict())
 import wandb; wr=wandb.init(project=a.wandb_project,name=a.wandb_run_name,config={"protocol":"evolutionary_meta_network_three_source_benchmark_diagnostic_alzheimer_validation","train_datasets":train_ids,"benchmark":valid,"target":alz,"trials":a.trials,"patience":a.patience})
 for step in range(start,a.trials):
  model=copy.deepcopy(base); model.load_state_dict(best_state)
  if step>0:
   with torch.no_grad():
    for q in model.parameters(): q.add_(torch.tensor(rng.normal(0,.08,size=tuple(q.shape)),dtype=q.dtype))
  model.eval()
  with torch.no_grad(): pred_b=_decode_direct(model.forward(zvalid),0,max_warmup,{}); cfg=_decode_direct(model.forward(zalz),0,max_warmup,{})
  ref=bests[valid].config; err=float(np.mean([float(pred_b.get(k)!=ref.get(k)) if not isinstance(pred_b.get(k,0),(int,float)) else abs(float(pred_b.get(k,0))-float(ref.get(k,0))) for k in ref]))
  X,y,b=data[alz]; run=argparse.Namespace(**vars(hp_search.parse_args([]))); run.dataset=alz; run.n_epochs=a.n_epochs; run.n_repeats=a.n_repeats; run.num_workers=a.num_workers; run.device=a.device; run.seed=a.seed+10000+step; run.results_dir=str(a.output_dir/"alzheimer"); run.cv_split_cache=str(a.output_dir/"cv_splits"/"massbench_alzheimer.npz"); run.resolved_n_repeats=hp_search.resolve_n_repeats(run.n_repeats,b)
  try: score,metrics=hp_search.run_trial(cfg,run,(X,y,b),f"meta_evolution_{step}_alzheimer",fixed_test_data=hp_search.load_fixed_test_dataset(alz))
  except Exception as e: score=-1.; metrics={"error":f"{type(e).__name__}: {e}"}
  score=float(score); improved=score>best_score
  if improved: best_score=score; stale=0; best_state=copy.deepcopy(model.state_dict()); torch.save(best_state,modelp)
  else: stale+=1
  universal_best, universal_is_best = update_best(a.output_dir.parent / "best_alzheimer.json", score=score, strategy="meta_evolution_alzheimer_validation", trial=step+1, config=cfg, run_name=a.wandb_run_name, output_dir=str(a.output_dir), extra={"benchmark_config_error": float(err)})
  rec={"trial":step+1,"benchmark_prediction":pred_b,"benchmark_config_error":err,"alzheimer_config":cfg,"alzheimer_valid_mcc":score,"alzheimer_metrics":metrics,"best_alzheimer_valid_mcc":best_score,"universal_best_alzheimer":universal_best,"stale_trials":stale,"improved":improved}
  with ledger.open("a") as f: f.write(json.dumps(rec,default=str)+chr(10)); f.flush(); os.fsync(f.fileno())
  state={"next_trial":step+1,"best_alzheimer_valid_mcc":best_score,"universal_best_alzheimer":universal_best,"stale_trials":stale,"trials":a.trials,"patience":a.patience,"best_model":str(modelp),"train_datasets":train_ids,"benchmark":valid,"target":alz}; statep.write_text(json.dumps(state,indent=2)+chr(10))
  wr.log({"leaderboard/best_alzheimer_valid_mcc":float(universal_best["score"]),"leaderboard/is_current_best":int(universal_is_best)},commit=False)
  wr.log({"trial":step+1,"benchmark_hparam_error":err,"alzheimer_valid_mcc":score,"best_alzheimer_valid_mcc":best_score,"universal_best_alzheimer":universal_best,"stale_trials":stale,**{"alzheimer/"+k:v for k,v in cfg.items() if isinstance(v,(int,float,bool))}})
  print(f"[meta-evolution] trial={step+1}/{a.trials} benchmark_error={err:.4f} Alzheimer={score:.4f} best={best_score:.4f} stale={stale}/{a.patience}",flush=True)
  if stale>=a.patience: break
 wr.finish()
if __name__=="__main__": main()
