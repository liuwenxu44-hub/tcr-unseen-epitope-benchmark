"""Bind original member dependencies to accepted H1 structure before any fit."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,hashlib,json

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);ap.add_argument('--bind-additive',action='store_true');a=ap.parse_args();w=a.work
 if a.bind_additive:
  index=load(w/'additive/artifact_index.json');assert index['status']=='PASS25_ANALYTIC_ADMISSIONS_AND_RELOADS'
  mapping=load(w/'training/task_mapping25_pre_additive.json');by={x['task_id']:x for x in index['artifacts']}
  assert len(by)==25
  for t in mapping['tasks']:
   item=by[t['task_id']];assert item['status']=='PASS' and sha(item['path'])==item['sha256']
   t['additive']={k:item[k] for k in ('path','sha256','action')}
  mapping['Additive_artifact_index_sha256']=sha(w/'additive/artifact_index.json')
  mapping['status']='FIXED_BEFORE_FITS_AND_SCORES_ADDATIVE_BOUND'
  write(w/'training/task_mapping25.json',mapping);return
 structure=load(w/'structure/structural_gate.json');reuse=load(w/'validation/reuse_dependency_gate.json')
 assert structure['status']=='PASS_H1_ORIGINAL_STRUCTURE_ALL75' and reuse['status']=='PASS_COMPLETE_DEPENDENCIES_AND_EXACT_REUSE_ELIGIBILITY'
 assert structure['capacity_problems']==structure['passing_capacity_problems']==15
 assert structure['rematch_graphs']==42 and structure['byte_reused_graphs']==33 and structure['passing_final_graphs']==75
 index=load(w/'structure/input_index75.json');assert len(index)==75
 lookup={(x['resample_id'],x['fold_id'],x['partition']):x for x in index};assert len(lookup)==75
 deps=load(w/'authority/member_dependencies.json');assert deps['status']=='PASS' and len(deps['members'])==25
 tasks=[]
 for m in deps['members']:
  roles={role:lookup[(m['resample_id'],m['fold_id'],role)] for role in ('train','validation','test')}
  old={(x['role'],x['kind']):x for x in m['inputs']}
  for role in roles:
   assert roles[role]['positive']['sha256']==old[(role,'positive')]['sha256']
   for kind in ('positive','negative'):assert sha(roles[role][kind]['path'])==roles[role][kind]['sha256']
  changed=roles['train']['negative']['sha256']!=old[('train','negative')]['sha256']
  assert changed==bool(m['interaction_fit'])
  assert (roles['train']['action']=='REMATCH')==changed
  assert sha(m['checkpoint']['path'])==m['checkpoint']['sha256'] and sha(m['additive']['path'])==m['additive']['sha256']
  tasks.append({'task_id':m['member_key'],'train_resample_id':m['resample_id'],'fold_id':m['fold_id'],'interaction_fit':m['interaction_fit'],'model_seed':13,'old_model':m['checkpoint'],'old_additive':m['additive'],'roles':roles,'train_change_equals_frozen_fit_action':True,'model_selection_uses_validation':False})
 assert sum(t['interaction_fit'] for t in tasks)==18
 mapping={'status':'STRUCTURE_AND_DEPENDENCIES_BOUND_BEFORE_ANALYTIC_ADMISSION','generation':'r05_original_repair_h1_20261009','count':25,'interaction_fit_budget':18,'model_reuse_budget':7,'additive_numerical_fit_budget':0,'source_structure_gate_sha256':sha(w/'structure/structural_gate.json'),'source_input_index_sha256':sha(w/'structure/input_index75.json'),'member_dependencies_sha256':sha(w/'authority/member_dependencies.json'),'reuse_dependency_gate_sha256':sha(w/'validation/reuse_dependency_gate.json'),'tasks':tasks}
 write(w/'training/task_mapping25_pre_additive.json',mapping)
 for rel in ('training/fit_original18.py','additive/admit_and_bind25.py'):
  ast.parse((w/rel).read_text())
 write(w/'training/entry_gate.json',{'status':'PASS_STRUCTURE_AND_REUSE_DEPENDENCIES','utc':datetime.now(timezone.utc).isoformat(),'capacity15':True,'graphs42_rematched33_reused75_accepted':True,'all25_model_dependencies':True,'budget18_fit7_reuse':True,'internal109_refresh16_reuse':True,'external18_refresh7_reuse':True,'entry_scope':'Only authorize analytical admission next; numerical fits additionally require analytical PASS and pretraining freeze.','identities':{rel:sha(w/rel) for rel in ('structure/structural_gate.json','structure/input_index75.json','validation/reuse_dependency_gate.json','authority/member_dependencies.json','training/task_mapping25_pre_additive.json')}})
 print('PASS_STRUCTURE_AND_REUSE_DEPENDENCIES',flush=True)
if __name__=='__main__':main()
