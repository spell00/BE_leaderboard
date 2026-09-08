#!/usr/bin/env python3
"""Three-source supervised meta-training, benchmark prediction-only validation, Alzheimer evaluation."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from scripts import hp_search
from src.meta_hpo_bank import load_bank, source_dataset_ids, best_by_dataset, trials_at_prefix
from src.meta_hpo_models import DirectBERNNMetaModel, normalize_source_meta, _direct_targets, _direct_loss, _decode_direct
from src.zero_shot_recommender.meta_features import extract_meta_features, META_FEATURE_NAMES
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--trial-bank",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True)
 p.add_argument("--validation-source",default="massbench_benchmark"); p.add_argument("--epochs",type=int,default=1000)
 p.add_argument("--hidden-size",type=int,default=64); p.add_argument("--lr",type=float,default=1e-2)
 p.add_argument("--n-epochs",type=int,default=1000); p.add_argument("--n-repeats",type=int,default=3)
 p.add_argument("--num-workers",type=int,default=4); p.add_argument("--device",default="cuda")
 p.add_argument("--seed",type=int,default=42); p.add_argument("--no-wandb",action="store_true")
 p.add_argument("--wandb-project",default="BE_leaderboard_meta_evolution"); p.add_argument("--wandb-run-name",default="meta-train3-benchmark-validate-alzheimer")
 a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
 trials=load_bank(a.trial_bank); sources=source_dataset_ids(trials)
 if a.validation_source not in sources: raise ValueError("validation source is not in bank")
 train_ids=tuple(x for x in sources if x != a.validation_source); alz="massbench_alzheimer"
 data={x:hp_search.load_dataset(x) for x in sources+(alz,)}; meta={}
 for x,(X,y,b) in data.items():
  f=extract_meta_features(X,y,b); meta[x]=np.asarray([f[n] for n in META_FEATURE_NAMES],dtype=np.float32)
 train_meta,mean,scale=normalize_source_meta(meta,train_ids); max_warmup=max(1,min(50,a.n_epochs))
 best=best_by_dataset(trials_at_prefix(trials,20,role="source"),sources); train_best={x:best[x] for x in train_ids}
 target_meta={x:(meta[x]-mean)/scale for x in (a.validation_source,alz)}
 torch.manual_seed(a.seed); model=DirectBERNNMetaModel(len(mean),a.hidden_size,{})
 opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-5); X=torch.tensor(np.stack([train_meta[x] for x in train_ids]),dtype=torch.float32); targets=_direct_targets(train_best,train_ids,max_warmup)
 history=[]; run=None
 if not a.no_wandb:
  import wandb; run=wandb.init(project=a.wandb_project,name=a.wandb_run_name,config={"train_datasets":train_ids,"validation_source":a.validation_source,"target":alz,"bank":str(a.trial_bank),"epochs":a.epochs})
 model.train()
 for epoch in range(1,a.epochs+1):
  opt.zero_grad(set_to_none=True); out=model.forward(X); loss=_direct_loss(out,targets,model.fixed); loss.backward(); opt.step(); model.eval()
  with torch.no_grad():
   def pred(name): return _decode_direct(model.forward(torch.tensor(target_meta[name][None,:],dtype=torch.float32)),0,max_warmup,{})
   vb=pred(a.validation_source); az=pred(alz)
  ref=best[a.validation_source].config
  err=float(np.mean([abs(float(vb.get(k,0))-float(ref.get(k,0))) if isinstance(vb.get(k,0),(int,float)) else float(vb.get(k)!=ref.get(k)) for k in ref]))
  rec={"epoch":epoch,"train_loss":float(loss.detach()),"benchmark_predicted_config":vb,"benchmark_config_error":err,"alzheimer_predicted_config":az}; history.append(rec)
  if run: run.log({"epoch":epoch,"train_loss":rec["train_loss"],"benchmark_config_error":err,**{"benchmark/"+k:v for k,v in vb.items() if isinstance(v,(int,float,bool))}})
  model.train()
 cfg=history[-1]["alzheimer_predicted_config"]; Xd,yd,bd=data[alz]; hp=hp_search.parse_args([]); hp.n_epochs=a.n_epochs; hp.n_repeats=a.n_repeats; hp.num_workers=a.num_workers; hp.device=a.device; hp.dataset=alz; hp.seed=a.seed; hp.results_dir=str(a.output_dir/"alzheimer"); hp.cv_split_cache=str(a.output_dir/"cv_splits"/"massbench_alzheimer.npz"); hp.resolved_n_repeats=hp_search.resolve_n_repeats(hp.n_repeats,bd)
 score,metrics=hp_search.run_trial(cfg,hp,(Xd,yd,bd),"meta_train3_benchmark_validate_alzheimer")
 result={"protocol":"train_three_sources_validate_benchmark_predict_only_then_alzheimer","train_datasets":train_ids,"benchmark":a.validation_source,"alzheimer":alz,"epochs":a.epochs,"best_benchmark_bank_config":best[a.validation_source].config,"benchmark_final_prediction":history[-1]["benchmark_predicted_config"],"benchmark_config_error":history[-1]["benchmark_config_error"],"alzheimer_config":cfg,"alzheimer_valid_mcc":float(score),"alzheimer_metrics":metrics}
 (a.output_dir/"result.json").write_text(json.dumps(result,indent=2,default=str)+chr(10)); (a.output_dir/"training_history.jsonl").write_text(chr(10).join(json.dumps(r,default=str) for r in history)+chr(10))
 if run: run.log({"alzheimer_valid_mcc":float(score)}); run.finish()
 print(json.dumps({"alzheimer_valid_mcc":float(score),"benchmark_config_error":history[-1]["benchmark_config_error"]}))
if __name__=="__main__": main()
