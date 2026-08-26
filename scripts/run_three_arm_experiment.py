#!/usr/bin/env python3
"""Resume-safe sequential launcher for the three 30-candidate HPO arms."""
import argparse, json, subprocess, sys, time, uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--resume",action="store_true")
    p.add_argument("--n-epochs",type=int,default=1000); p.add_argument("--n-repeats",type=int,default=3)
    p.add_argument("--batch-size",type=int,default=32); p.add_argument("--num-workers",type=int,default=4)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda"); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--no-wandb",action="store_true"); p.add_argument("--smoke",action="store_true")
    p.add_argument("--plan",action="store_true",help="Print commands without creating state or running arms")
    return p.parse_args(argv)
def atomic(path,obj):
    t=path.with_suffix(".tmp"); t.write_text(json.dumps(obj,indent=2)+"\n"); t.replace(path)
def main(argv=None):
    a=parse_args(argv); a.output_dir.mkdir(parents=True,exist_ok=True); statep=a.output_dir/"orchestrator_state.json"
    if a.n_repeats != 3:
        raise ValueError("The three-arm comparison requires grouped CV=3 for every arm")
    if statep.exists() and not a.resume: raise FileExistsError("pass --resume")
    state=json.loads(statep.read_text()) if statep.exists() else {"experiment_id":uuid.uuid4().hex[:8],"arms":{}}
    common=["--n-epochs",str(a.n_epochs),"--n-repeats",str(a.n_repeats),"--device",a.device,"--seed",str(a.seed)]
    arms=[
      ("meta_evolution",[sys.executable,str(ROOT/"scripts/evolve_meta_model.py"),"--population-size","6","--generations","5","--elite-count","2","--tournament-size","3"]),
      ("dataset_conditional",[sys.executable,str(ROOT/"scripts/run_optuna_comparison.py"),"--n-trials","30","--meta-hidden-size","64"]),
      ("global_shared",[sys.executable,str(ROOT/"scripts/run_global_shared_optuna.py"),"--n-trials","30"])]
    if a.plan:
        for name,cmd in arms:
            print(name, " ".join(cmd))
        return 0
    for name,cmd in arms:
        rec=state["arms"].setdefault(name,{"status":"pending","wandb_run_name":f"three-arm-{state['experiment_id']}-{name}"})
        if rec["status"]=="succeeded": continue
        out=a.output_dir/name; cmd += ["--output-dir",str(out),"--wandb-run-name",rec["wandb_run_name"],*common]
        if name!="meta_evolution": cmd += ["--batch-size",str(a.batch_size),"--num-workers",str(a.num_workers)]
        else: cmd += ["--num-workers",str(a.num_workers)]
        if a.no_wandb: cmd.append("--no-wandb")
        resume_markers = {
            "meta_evolution": (out / "checkpoint.npz", out / "state.json"),
            "dataset_conditional": (out / "run_metadata.json", out / "optuna.sqlite3"),
            "global_shared": (out / "run_metadata.json", out / "optuna.sqlite3"),
        }
        # An orchestrator retry is not necessarily an arm resume: preflight can
        # fail before the arm creates any checkpoint. Forward --resume only when
        # the arm has all of its own durable resume artifacts.
        if a.resume and all(path.exists() for path in resume_markers[name]):
            cmd.append("--resume")
        if a.smoke:
            cmd += ["--smoke-evaluator"] if name=="meta_evolution" else (["--smoke"] if name=="global_shared" else [])
            if name=="dataset_conditional": raise ValueError("dataset-conditional arm has no smoke evaluator; use its tests")
        rec.update(status="running",command=cmd,started_at=time.time()); atomic(statep,state)
        result=subprocess.run(cmd,cwd=ROOT)
        rec.update(status="succeeded" if result.returncode==0 else "failed",returncode=result.returncode,finished_at=time.time()); atomic(statep,state)
        if result.returncode: return result.returncode
    return 0
if __name__=="__main__": raise SystemExit(main())
