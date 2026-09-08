"""Shared cross-strategy Alzheimer leaderboard with atomic updates."""
from __future__ import annotations
import json, os, time
from pathlib import Path
import fcntl
def update_best(path: str | Path, *, score: float, strategy: str, trial: int, config: dict, run_name: str | None = None, output_dir: str | None = None, extra: dict | None = None) -> tuple[dict, bool]:
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); lock_path=path.with_suffix(path.suffix+".lock")
 with lock_path.open("a+") as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
  try:
   current={}
   if path.exists():
    try: current=json.loads(path.read_text())
    except (OSError,json.JSONDecodeError): current={}
   candidate={"score":float(score),"strategy":strategy,"trial":int(trial),"config":config,"run_name":run_name,"output_dir":output_dir,"updated_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),**(extra or {})}
   improved=not current or float(score)>float(current.get("score",-float("inf")))
   if improved:
    tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(candidate,indent=2,default=str)+chr(10)); os.replace(tmp,path); current=candidate
   return current,improved
  finally: fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
