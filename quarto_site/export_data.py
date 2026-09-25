"""Read-only snapshot exporter for the parallel Quarto frontend."""
from __future__ import annotations
import json, math, sqlite3
from pathlib import Path
SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
LABELS = {"bacteria_2024_mz10":"Urinary Pathogens · LC-MS/MS","colon_3041":"Colon 3041","jdlber_sle_maldi":"SLE · MALDI","massbench_adenocarcinoma":"MassBench Adenocarcinoma","massbench_alzheimer":"MassBench Alzheimer","massbench_benchmark":"MassBench Benchmark","normal_tissue_878":"Normal Tissue 878","scib_pancreas":"scIB Pancreas","seqc_maqc":"SEQC / MAQC"}
con = sqlite3.connect(ROOT / "data" / "leaderboard.db")
con.row_factory = sqlite3.Row
sql = """select s.id submission_id,s.username,s.dataset,s.submission_name,s.is_public,s.created_at,
sc.valid_mcc,sc.test_mcc,sc.accuracy,sc.macro_f1,sc.evaluation_protocol,sc.cv_folds,sc.source_file,sc.version_evaluated
from submissions s join scores sc on sc.submission_id=s.id"""
rows = []
for raw in con.execute(sql):
    r = dict(raw)
    valid = float(r["valid_mcc"]) if r["valid_mcc"] is not None and math.isfinite(float(r["valid_mcc"])) else 0.0
    test = float(r["test_mcc"]) if r["test_mcc"] is not None and math.isfinite(float(r["test_mcc"])) else 0.0
    r["score"] = min(valid, test)
    rows.append(r)
rows.sort(key=lambda r:(r["score"], r["accuracy"] or 0.0), reverse=True)
keys = sorted(p.name for p in (ROOT / "data" / "datasets").iterdir() if p.is_dir())
datasets = [{"key":k,"label":LABELS.get(k,k.replace("_"," ").title()),"runs":sum(r["dataset"]==k for r in rows),"protocols":sorted({r["evaluation_protocol"] for r in rows if r["dataset"]==k and r["evaluation_protocol"]})} for k in keys]
payload = {"stats":{"datasets":len(datasets),"runs":len(rows),"protocols":len({r["evaluation_protocol"] for r in rows if r["evaluation_protocol"]}),"contributors":len({r["username"] for r in rows if r["username"]})},"datasets":datasets,"leaderboard":rows}
out = SITE / "data" / "site-data.js"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("window.BE_SITE_DATA = " + json.dumps(payload, indent=2, default=str) + ";\n", encoding="utf-8")
print(f"[quarto] exported {len(rows)} leaderboard rows and {len(datasets)} datasets")
