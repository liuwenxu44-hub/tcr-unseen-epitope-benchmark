"""Select and freeze the explicit methods/input lineage before numerical fitting."""
from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json, subprocess, sys

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def new(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False);f.write('\n')

def main():
    w=Path(sys.argv[1]).resolve();gate=Path(sys.argv[2]).resolve()
    assert not any((w/'execution').iterdir()), 'No freeze after numerical execution begins'
    model=read(w/'authority/model_contract.json');tasks=read(w/'authority/task_mapping105.json')['tasks']
    artifacts=read(w/'additive/artifact_index.json')['artifacts']
    assert len(tasks)==len(artifacts)==105 and all(a['status']=='PASS' for a in artifacts)
    assert {a['task_id'] for a in artifacts}=={t['task_id'] for t in tasks}
    assert read(w/'metrics/preperformance_metric_selfcheck.json')['status']=='PASS'
    tree=ast.parse((w/'run_g2_training.py').read_text())
    fit_calls=[ast.unparse(x) for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='fit']
    assert fit_calls==["model.fit(matrices['train'], labels)"],fit_calls
    fixed={
      'METHOD_EXECUTOR':['run_g2_training.py','launch_g2_training.sh','freeze_pretraining.py','training_protocol.json','authority/freeze_manifest.py'],
      'INPUT_AUTHORITY':['authority/task_mapping105.json','authority/task_mapping105.csv','authority/model_contract.json','authority/intake_summary.json','authority/g2_manifest_before.json','authority/g2_manifest_parent.json','authority/protected_identity_before.json','authority/revision_plan_20261008_v1_1.md','authority/execution_contract.json','authority/final_allocations.json','authority/release_decision.json','authority/r01_static_feature_objective_audit.json'],
      'ANALYTIC_METHOD':['audit_additive_g2.py','audit_additive_shared_transform.py','audit_additive_launch.py','additive/protocol.json','additive/admission_index.json','additive/admission_table.csv','additive/artifact_index.json'],
      'METRIC_METHOD':['summarize_r04_g2.py','validate_r04_metrics_g2.py','metrics/calculation_contract.json','metrics/code_contract_manifest.json','metrics/preperformance_metric_selfcheck.json','metrics/preperformance_failure_propagation_check.json'],
      'INDEPENDENT_NO_FIT_VALIDATION_CODE':['validation/model_roundtrip105.py'],
    }
    fixed['SOURCE_SNAPSHOT']=['authority/source_snapshot/'+x['relative_path'] for x in model['codeidentities']+model['config_files']]
    fixed['ANALYTIC_MODEL']=[x['artifact_file'] for x in artifacts]
    # These are explicit task-owned admission records, selected by the closed task list.
    fixed['ANALYTIC_ADMISSION']=['additive/admissions/'+t['task_id']+'.json' for t in tasks]
    files=[]
    for role,paths in fixed.items():
        for p in paths:
            q=w/p;assert q.is_file(),str(q)
            files.append({'role':role,'path':p,'bytes':q.stat().st_size,'sha256':sha(q)})
    assert len(files)==len({x['path'] for x in files})
    new(w/'pretraining_identity.json',{'status':'FROZEN_BEFORE_NUMERICAL_FITS','utc':datetime.now(timezone.utc).isoformat(),'g2_manifest_sha256':'98bfca307c68f01f6980f3f2f0d6bf01529254cd554ffff4a253d6fc54942d35','numerical_fit_callsite':fit_calls,'files':files})
    args=[sys.executable,str(gate),'create','--root',str(w),'--output','pretraining_manifest.json','--analysis-id','R04_G2_105_TRAINING_METHODS_20261009','--status','frozen','--allow-dirty','--note','Explicit methods-only release. No numerical interaction fit or new performance has been computed. Parent data selected solely via G2 manifest. Existing project Git dirtiness is documented, not reset.']
    for x in files+[{'role':'PRETRAINING_IDENTITY','path':'pretraining_identity.json'}]:args+=['--file',x['role']+'='+x['path']]
    subprocess.run(args,check=True)
    subprocess.run([sys.executable,str(gate),'check','--root',str(w),'--manifest',str(w/'pretraining_manifest.json'),'--report',str(w/'validation/pretraining_manifest_check.json')],check=True,stdout=subprocess.DEVNULL)
    print(json.dumps({'status':'FROZEN_BEFORE_FITS','files':len(files),'manifest_sha256':sha(w/'pretraining_manifest.json')}))
if __name__=='__main__':main()
