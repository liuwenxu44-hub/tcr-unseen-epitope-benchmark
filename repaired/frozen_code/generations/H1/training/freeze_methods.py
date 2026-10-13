"""Freeze explicit H1 inputs and methods before all numerical fits/scores."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,subprocess,sys
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
 w=Path(sys.argv[1]);assert not(w/'training/tasks').exists() and not(w/'scoring').exists()
 assert load(w/'training/entry_gate.json')['status']=='PASS_STRUCTURE_AND_REUSE_DEPENDENCIES'
 assert load(w/'additive/artifact_index.json')['status']=='PASS25_ANALYTIC_ADMISSIONS_AND_RELOADS'
 assert load(w/'validation/reuse_historical_code_gate.json')['status']=='PASS_HISTORICAL_PIPELINE_CODE_IDENTITIES'
 assert load(w/'metrics/synthetic_formula_checks.json')['status']=='PASS_SYNTHETIC_FORMULA_AND_AGGREGATION_ONLY'
 assert load(w/'validation/preexecution_code_review.json')['status']=='PASS_STATIC_PREEXECUTION_CODE_REVIEW'
 assert load(w/'validation/metrics_preexecution_review.json')['status']=='PASS_METRICS_STATIC_PREEXECUTION_REVIEW'
 selected={}
 def add(rel,role,expected=None):
  p=w/rel;assert p.is_file(),rel
  x={'role':role,'path':rel,'bytes':p.stat().st_size,'sha256':sha(p)}
  if expected:assert x['sha256']==expected['sha256'] and x['bytes']==expected['bytes']
  if rel in selected:assert selected[rel]['sha256']==x['sha256']
  selected[rel]=x
 for name,prefix in [('structure/owned_catalog.json','structure/'),('metrics/preperformance_owned_catalog.json','')]:
  for item in load(w/name)['files']:add(prefix+item['path'],item.get('role','METHOD_OR_INPUT'),item)
  add(name,'EXPLICIT_PRODUCER_CATALOG')
 fixed=['scope_amendment.json','training/fit_original18.py','training/prepare_original25.py','training/attempt1_prepare_original25.py','training/launch_h1_phase.sh','training/training_protocol.json','training/freeze_methods.py','training/task_mapping25_pre_additive.json','training/task_mapping25.json','training/entry_gate.json','additive/admit_and_bind25.py','additive/rejection_regressions.json','additive/admission_index.json','additive/admission_table.csv','additive/artifact_index.json','authority/freeze_manifest.py','authority/revision_plan_20261008_v1_1.md','authority/audit_original_dependencies.py','authority/audit_historical_code_identity.py','authority/model_contract.json','authority/member_dependencies.json','authority/internal_prediction_sources.json','authority/external_prediction_sources.json','validation/reuse_dependency_gate.json','validation/reuse_historical_code_gate.json','validation/preexecution_code_review.json','validation/metrics_preexecution_review.json','validation/additive_independent25.py','validation/additive_independent25.json','validation/model_entity25.py']
 for rel in fixed:add(rel,'H1_METHOD_OR_ADMISSION_AUTHORITY')
 for item in load(w/'authority/model_contract.json')['source_files']:add(item['snapshot_path'],'ORIGINAL_SOURCE_CONFIG_ENV_SNAPSHOT',item)
 for t in load(w/'training/task_mapping25.json')['tasks']:
  for prefix,suffix in [('additive/admissions/','.json'),('additive/node_counts/','.csv.gz')]:add(prefix+t['task_id']+suffix,'H1_TASK_ANALYTIC_ADMISSION')
 for a in load(w/'additive/artifact_index.json')['artifacts']:
  if a['action']=='NEW_INPUT_BOUND_ANALYTIC_ARTIFACT':add(Path(a['path']).relative_to(w).as_posix(),'NEW_INPUT_BOUND_ANALYTIC_MODEL')
 tree=ast.parse((w/'training/fit_original18.py').read_text());calls=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='fit'];assert calls==['model.fit(X, labels)']
 review=load(w/'validation/preexecution_code_review.json')
 # Review includes exact reviewed source SHA checks; a subsequent edit is blocked.
 assert sha(w/'training/fit_original18.py')=='f9a843683523ac712163d73291dc79c59e5213075bd90c1a416f2ea73b5175b2'
 assert sha(w/'additive/admit_and_bind25.py')=='f91d876209a65c7bfa9e8e470a7fe7c8bb69b40f862dc4d5ea65dfa708e07489'
 write(w/'training/pretraining_identity.json',{'status':'FROZEN_BEFORE_FITS_AND_SCORES','utc':datetime.now(timezone.utc).isoformat(),'generation':'r05_original_repair_h1_20261009','numerical_fit_callsite':calls,'interaction_fits':18,'Additive_numerical_fits':0,'scientific_state':'posthoc original-preserving repair; no new external validation','files':list(selected.values())})
 add('training/pretraining_identity.json','PRETRAINING_IDENTITY')
 args=[sys.executable,str(w/'authority/freeze_manifest.py'),'create','--root',str(w),'--output','training/pretraining_manifest.json','--analysis-id','R05_H1_ORIGINAL_REPAIR_METHODS_20261009','--status','frozen','--allow-dirty','--note','Authorized independent H1 repair generation. Explicit methods and accepted original-preserving inputs before18 shared fits/109+18 score tasks;0 new inferential analyses.']
 for item in selected.values():args+=['--file',item['role']+'='+item['path']]
 subprocess.run(args,check=True)
 subprocess.run([sys.executable,str(w/'authority/freeze_manifest.py'),'check','--root',str(w),'--manifest',str(w/'training/pretraining_manifest.json'),'--report',str(w/'validation/pretraining_manifest_server_check.json')],check=True,stdout=subprocess.DEVNULL)
 print(json.dumps({'status':'FROZEN_BEFORE_FITS_AND_SCORES','files':len(selected),'manifest_sha256':sha(w/'training/pretraining_manifest.json')}),flush=True)
if __name__=='__main__':main()
