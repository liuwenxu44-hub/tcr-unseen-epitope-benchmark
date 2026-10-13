"""Fixed isolated CPU launch for pure Additive admission; never optimizer fit."""
import argparse
import os
import subprocess
from pathlib import Path

BASE = Path("/data/lwx/proj/TCR_unseen_epitope/runs/revision_20261008_v1_1")
WORK = BASE / "r04_training_g2_20261009"
IMAGE = "sha256:d7c8838e67a156a9ed1abe981fee56f55a3637f2af47bae7b256a3c3c7551b59"

parser = argparse.ArgumentParser()
parser.add_argument("--prepare", action="store_true")
parser.add_argument("--reconcile", action="store_true")
args = parser.parse_args()
if args.prepare and args.reconcile:
    raise ValueError("choose one launch mode")
mode = "prepare" if args.prepare else ("reconcile" if args.reconcile else "admission")
script = "/work/audit_additive_shared_transform.py" if args.reconcile else "/work/audit_additive_g2.py"
command = ["docker", "run", "--rm", "--read-only", "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}", "--entrypoint", "python",
           "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONUNBUFFERED=1", "-e", "PYTHONPATH=/source:/work", "-e", "OMP_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1", "-e", "TMPDIR=/work/scratch", "-e", "XDG_CACHE_HOME=/work/scratch",
           "-v", f"{WORK}:/work:rw", "-v", f"{WORK / 'authority/source_snapshot'}:/source:ro", "-v", f"{BASE / 'r04_structure_g2_20261009'}:/g2:ro", "-v", f"{BASE / 'r01'}:/r01:ro", "-w", "/work", IMAGE,
           script, "--work", "/work"]
if not args.reconcile:
    command.extend(["--source", "/source", "--g2", "/g2", "--r01", "/r01"])
if args.prepare:
    command.append("--prepare")
with (WORK / "additive" / (mode + "_stdout.log")).open("x", encoding="utf-8") as log:
    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
print("Additive", mode, "exit:", process.returncode, flush=True)
raise SystemExit(process.returncode)
