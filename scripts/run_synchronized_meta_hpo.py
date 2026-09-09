#!/usr/bin/env python3
"""Synchronized four-dataset HPO with benchmark-selected direct meta-learning."""
import argparse,json,os,sys,time,uuid
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(ROOT))
from scripts import hp_search
from src.meta_hpo_utils import sample_bernn_config
from src.meta_hpo_bank import BankTrial,config_feature_vector
from src.meta_hpo_models import train_direct_meta_model,normalize_source_meta,_decode_direct
from src.meta_leaderboard import update_best
from src.zero_shot_recommender.meta_features import extract_meta_features,META_FEATURE_NAMES
def payload(t):
 a=dict(t.user_attrs); return {"trial_number":int(t.number),"valid_mcc":float(t.value),"test_mcc":float(a.get("test_mcc",np.nan)),"fit_seconds":float(a.get("fit_seconds",np.nan)),"config":a["config"],"valid_mcc_folds":tuple(a.get("valid_mcc_folds",())),"test_mcc_folds":tuple(a.get("test_mcc_folds",())),"error":a.get("error")}
def all_dataset_telemetry(current, best):
    telemetry = {}
    for dataset_id, trial in current.items():
        for label, row in (("current", trial), ("best", best[dataset_id])):
            prefix = f"dataset/{dataset_id}/{label}"
            for metric in ("valid_mcc", "test_mcc", "fit_seconds"):
                value = row.get(metric)
                if isinstance(value, (int, float, np.integer, np.floating)):
                    telemetry[f"{prefix}/{metric}"] = float(value)
            for name, value in row.get("config", {}).items():
                if isinstance(value, (bool, int, float, np.integer, np.floating)):
                    telemetry[f"{prefix}/hparams/{name}"] = float(value)
                elif isinstance(value, str):
                    telemetry[f"{prefix}/hparams/{name}"] = value
    return telemetry

def main():
 p=argparse.ArgumentParser(); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--n-trials",type=int,default=100); p.add_argument("--n-epochs",type=int,default=1000); p.add_argument("--n-repeats",type=int,default=3); p.add_argument("--batch-size",type=int,default=32); p.add_argument("--num-workers",type=int,default=4); p.add_argument("--device",default="cuda"); p.add_argument("--seed",type=int,default=42); p.add_argument("--resume",action="store_true"); p.add_argument("--no-wandb",action="store_true"); p.add_argument("--wandb-project",default="BE_leaderboard_meta_evolution"); p.add_argument("--wandb-run-name",default="synchronized-meta-hpo-benchmark-selected")
 p.add_argument("--wandb-id",default=None); a=p.parse_args()
 import optuna
 a.output_dir.mkdir(parents=True,exist_ok=True); datasets=("normal_tissue_878","colon_3041","massbench_adenocarcinoma","massbench_benchmark"); train_ids=datasets[:3]; valid_id=datasets[3]; alz="massbench_alzheimer"
 data={d:hp_search.load_dataset(d) for d in datasets+(alz,)}; meta={}
 for d,(X,y,b) in data.items():
  f=extract_meta_features(X,y,b); meta[d]=np.asarray([f[n] for n in META_FEATURE_NAMES],dtype=np.float32)
 storage=optuna.storages.RDBStorage(url=f"sqlite:///{(a.output_dir/'optuna.sqlite3').resolve()}"); studies={d:optuna.create_study(study_name="sync_"+d,direction="maximize",sampler=optuna.samplers.TPESampler(seed=a.seed),storage=storage,load_if_exists=True) for d in datasets}
 ledger=a.output_dir/"rounds.jsonl"; old=[json.loads(x) for x in ledger.open()] if a.resume and ledger.exists() else []; done={int(x["round"]) for x in old}; hp=hp_search.parse_args([]); hp.n_epochs=a.n_epochs; hp.n_repeats=a.n_repeats; hp.num_workers=a.num_workers; hp.device=a.device
 import wandb; wr=None if a.no_wandb else wandb.init(project=a.wandb_project,name=a.wandb_run_name,id=a.wandb_id,resume="allow" if a.wandb_id else None,config={"protocol":"synchronized_four_dataset_hpo_benchmark_meta_selection","train_datasets":datasets,"meta_train_datasets":train_ids,"meta_validation_dataset":valid_id,"target_dataset":alz,"n_trials":a.n_trials})
 for step in range(a.n_trials):
  current={}
  for j,d in enumerate(datasets):
   complete=[t for t in studies[d].trials if t.state==optuna.trial.TrialState.COMPLETE and t.value is not None]
   if len(complete)<=step:
    t=studies[d].ask(); X,y,b=data[d]; run=argparse.Namespace(**vars(hp)); run.dataset=d; run.seed=a.seed+step*10+j; run.bs=min(a.batch_size,max(1,len(y))); run.results_dir=str(a.output_dir/d); run.cv_split_cache=str(a.output_dir/"cv_splits"/(d+".npz")); run.resolved_n_repeats=hp_search.resolve_n_repeats(run.n_repeats,b)
    cfg=sample_bernn_config(t,run,{}); started=time.monotonic(); err=None; metrics={}
    try: score,metrics=hp_search.run_trial(cfg,run,(X,y,b),f"sync_meta_{step}_{d}",fixed_test_data=hp_search.load_fixed_test_dataset(d))
    except Exception as e: score=-1.; err=f"{type(e).__name__}: {e}"
    t.set_user_attr("config",cfg); t.set_user_attr("test_mcc",float(metrics.get("test_mcc",np.nan))); t.set_user_attr("fit_seconds",time.monotonic()-started); t.set_user_attr("error",err); studies[d].tell(t,float(score)); complete=[z for z in studies[d].trials if z.state==optuna.trial.TrialState.COMPLETE and z.value is not None]
   current[d]=payload(complete[step])
  if step in done: continue
  best={d:max([payload(t) for t in studies[d].trials if t.state==optuna.trial.TrialState.COMPLETE and t.value is not None],key=lambda r:r["valid_mcc"]) for d in datasets}
  dataset_telemetry = all_dataset_telemetry(current, best)
  source_meta,mean,scale=normalize_source_meta(meta,train_ids); zmeta={d:(meta[d]-mean)/scale for d in (valid_id,alz)}; candidates=[]
  for h,ml in ((32,1e-3),(64,3e-3),(128,1e-2)):
   bt={d:BankTrial(dataset_id=d,role="source",trial_index=step,optuna_trial_number=best[d]["trial_number"],valid_mcc=best[d]["valid_mcc"],test_mcc=best[d]["test_mcc"],fit_seconds=best[d]["fit_seconds"],config=best[d]["config"]) for d in train_ids}
   model,hist,diag=train_direct_meta_model(bt,source_meta,meta[alz],max_warmup=max(1,min(50,a.n_epochs)),hidden_size=h,epochs=200,lr=ml,seed=a.seed+step)
   with __import__("torch").no_grad(): pred=_decode_direct(model.forward(__import__("torch").tensor(zmeta[valid_id][None,:],dtype=__import__("torch").float32)),0,max(1,min(50,a.n_epochs)),{})
   err=float(np.linalg.norm(config_feature_vector(pred,max_warmup=max(1,min(50,a.n_epochs)))-config_feature_vector(best[valid_id]["config"],max_warmup=max(1,min(50,a.n_epochs)))))
   candidates.append((err,h,ml,model,diag))
  valerr,h,ml,model,diag=min(candidates,key=lambda x:x[0]); model.eval()
  with __import__("torch").no_grad(): alzcfg=_decode_direct(model.forward(__import__("torch").tensor(zmeta[alz][None,:],dtype=__import__("torch").float32)),0,max(1,min(50,a.n_epochs)),{})
  X,y,b=data[alz]; run=argparse.Namespace(**vars(hp)); run.dataset=alz; run.seed=a.seed+10000+step; run.results_dir=str(a.output_dir/"alzheimer"); run.cv_split_cache=str(a.output_dir/"cv_splits"/"massbench_alzheimer.npz"); run.resolved_n_repeats=hp_search.resolve_n_repeats(run.n_repeats,b)
  try: score,metrics=hp_search.run_trial(alzcfg,run,(X,y,b),f"sync_meta_alzheimer_{step}",fixed_test_data=hp_search.load_fixed_test_dataset(alz))
  except Exception as e: score=-1.; metrics={"error":f"{type(e).__name__}: {e}"}
  universal_best, universal_is_best = update_best(a.output_dir.parent / "best_alzheimer.json", score=float(score), strategy="synchronized_meta_hpo", trial=step+1, config=alzcfg, run_name=a.wandb_run_name, output_dir=str(a.output_dir), extra={"benchmark_hparam_error": float(valerr)})
  best_alz=max([float(x.get("alzheimer_valid_mcc",-1)) for x in old],default=-1); best_alz=max(best_alz,float(score)); rec={"round":step,"source_trials":[current[d] for d in datasets],"best_source":best,"meta_hidden_size":h,"meta_lr":ml,"benchmark_prediction_error":valerr,"benchmark_reference_config":best[valid_id]["config"],"alzheimer_config":alzcfg,"alzheimer_valid_mcc":float(score),"alzheimer_metrics":metrics,"best_alzheimer_valid_mcc":best_alz,"leaderboard/best_alzheimer_valid_mcc":float(universal_best["score"]),"leaderboard/is_current_best":int(universal_is_best),"universal_best_alzheimer":universal_best}
  with ledger.open("a") as f: f.write(json.dumps(rec,default=str)+chr(10)); f.flush(); os.fsync(f.fileno())
  if wr: wr.log({"round":step,"benchmark_hparam_error":valerr,"alzheimer_valid_mcc":float(score),"best_alzheimer_valid_mcc":best_alz,"leaderboard/best_alzheimer_valid_mcc":float(universal_best["score"]),"leaderboard/is_current_best":int(universal_is_best),"meta_hidden_size":h,"meta_lr":ml,**dataset_telemetry,**{"alzheimer/"+k:v for k,v in alzcfg.items() if isinstance(v,(int,float,bool))}})
  print(f"[sync-meta] round={step+1}/{a.n_trials} benchmark_error={valerr:.4f} Alzheimer={float(score):.4f}",flush=True); old.append(rec)
 if wr: wr.finish()
if __name__=="__main__": main()
