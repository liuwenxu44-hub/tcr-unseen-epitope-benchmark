"""Approve only the already budgeted H1 score tasks after independent gates."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,sys
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
 w=Path(sys.argv[1]);assert not(w/'scoring').exists()
 assert load(w/'training/fit_summary.json')['status']=='PASS18_SHARED_FITS'
 assert load(w/'training/model_index25.json')['status']=='PASS25_SHARED_INTERNAL_EXTERNAL_MEMBERS'
 assert load(w/'validation/model_entity25.json')['status']=='PASS25_INDEPENDENT_COLD_MODEL_ENTITIES_NO_EXTRA_SCORES'
 assert load(w/'validation/additive_independent25.json')['status']=='PASS25_INDEPENDENT_ACTUAL_INPUT_ADDITIVE_ENTITIES'
 assert load(w/'validation/postfit_method_manifest_check.json')['status']=='PASS'
 metric=w/'metrics/calculation_contract.json';plan=w/'metrics/scoring_plan.json'
 assert sha(metric)=='b569877deff083c799d88414a783bab3439fb5d346715f0c12e16fac4cf18281'
 assert sha(plan)=='ca07c5a610ee3b8cc9d9adac5255cbf676abc8f5015de759f13cc91a6be6d2b7'
 gate={'status':'PASS_H1_ALL75_GRAPH_AND25_MEMBER_ADMISSION','utc':datetime.now(timezone.utc).isoformat(),'metric_contract_sha256':sha(metric),'scoring_plan_sha256':sha(plan),'shared_members':25,'new_fits':18,'reused_models':7,'internal_new_score_budget':109,'internal_exact_reuses':16,'external_new_score_budget':18,'external_exact_reuses':7,'Additive_numerical_fits':0,'P_and_CI_cap':0,'inference_status':'NOT_INFERENTIAL','same_external_cohort_posthoc_repair':True,'source_gates':{rel:sha(w/rel) for rel in ('structure/structural_gate.json','training/pretraining_manifest.json','training/fit_summary.json','training/model_index25.json','additive/artifact_index.json','validation/additive_independent25.json','validation/model_entity25.json','validation/postfit_method_manifest_check.json','validation/reuse_dependency_gate.json','validation/reuse_historical_code_gate.json')}}
 write(w/'delivery/scoring_release.json',gate)
 files=['training/release_scoring.py','delivery/scoring_release.json','metrics/scoring_plan.json','training/pretraining_manifest.json','training/fit_summary.json','training/fit_ledger.json','training/model_index25.json','validation/model_entity25.json','validation/postfit_method_manifest_check.json']
 for r in load(w/'training/fit_ledger.json'):
  files += ['training/tasks/'+r['task_id']+'/'+name for name in ('model.joblib','model_metadata.json','fit_attempt.json','task_result.json')]
 args=[sys.executable,str(w/'authority/freeze_manifest.py'),'create','--root',str(w),'--output','training/scoring_methods_manifest.json','--analysis-id','R05_H1_SCORING_METHOD_AND_MODEL_IDENTITY_20261009','--status','frozen','--allow-dirty','--note','Exact109 internal/18 external refresh slots plus16/7 verified reuse. Shared25members after accepted75inputs and independent model/analytic checks. Same external cohort posthoc, noP/CI.']
 for rel in files:args+=['--file','H1_SCORING_BOUNDARY='+rel]
 subprocess.run(args,check=True)
 print(json.dumps({'status':gate['status'],'scoring_manifest_sha256':sha(w/'training/scoring_methods_manifest.json')}),flush=True)
if __name__=='__main__':main()
