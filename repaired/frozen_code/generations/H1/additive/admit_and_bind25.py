"""H1 pure marginal admission: 18 new analytic artifacts, 7 verified reuses."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,importlib.util,io,json,sys
from pathlib import Path
import joblib,numpy as np
from sklearn.linear_model import LogisticRegression

R01_SHA='5c0b0ab330f0c3b97d218846fd5fbf23b6aed2a2dc8504fa0b9f523742620ce8'
API_SHA='3bfcaab5cb923c300637e83100307c47c8b55826f675587a56bb783d3ae4f744'
STATIC_SHA='fe08f78dd0fc2cfc1d553979ff115a0ff53678f7cfb75d89be0896ebcf17b2d9'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(entry):
 p=Path(entry['path']);assert sha(p)==entry['sha256']
 op=gzip.open if p.suffix=='.gz' else open
 with op(p,'rt',encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 assert len(rows)==entry['rows'];return rows
def denied_fit(*a,**k):raise RuntimeError('Numerical fit prohibited in H1 Additive admission')

def main():
 ap=argparse.ArgumentParser()
 for k in ('work','r01','source'):ap.add_argument('--'+k,type=Path,required=True)
 a=ap.parse_args();w=a.work
 assert sha(a.r01/'r01_manifest.json')==R01_SHA and sha(a.r01/'r01_audit.py')==API_SHA and sha(a.r01/'static_feature_objective_audit.json')==STATIC_SHA
 static=load(a.r01/'static_feature_objective_audit.json')
 for rel,item in static['feature_code_contracts'].items():assert sha(a.source/rel)==item['sha256']
 entry=load(w/'training/entry_gate.json');assert entry['status']=='PASS_STRUCTURE_AND_REUSE_DEPENDENCIES'
 mapping=load(w/'training/task_mapping25_pre_additive.json');tasks=mapping['tasks'];contract=load(w/'authority/model_contract.json')
 assert len(tasks)==25 and sum(t['interaction_fit'] for t in tasks)==18
 spec=importlib.util.spec_from_file_location('h1_r01_api',a.r01/'r01_audit.py');api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
 from src.interaction_resample_robustness import _fixed_rows,_model_from_config
 from src.degree_matched_additive_logistic import build_additive_feature_matrix,additive_feature_dimensions
 from src.kmer_logistic import fit_hla_mapping
 params=_model_from_config(contract['model_config'],13).get_params(deep=False);assert params==contract['actual_estimator_params']
 LogisticRegression.fit=denied_fit
 evidence={'identity':True,'train_only':True,'static_feature_contract':True,'actual_weight_rule_verified':True,'actual_fit_call_verified':True}
 fixture=[{'tcr_cdr3_beta':'CASSA','epitope':'GILG','hla':'HLA-A*02:01','pmhc_id':'GILG|HLA-A*02:01','label':1},{'tcr_cdr3_beta':'CASSC','epitope':'NLVP','hla':'HLA-B*07:02','pmhc_id':'NLVP|HLA-B*07:02','label':1},{'tcr_cdr3_beta':'CASSA','epitope':'NLVP','hla':'HLA-B*07:02','pmhc_id':'NLVP|HLA-B*07:02','label':0},{'tcr_cdr3_beta':'CASSC','epitope':'GILG','hla':'HLA-A*02:01','pmhc_id':'GILG|HLA-A*02:01','label':0}]
 builder=lambda rr:build_additive_feature_matrix(rr,fit_hla_mapping(rr))
 base,_=api.admit(fixture,[1]*4,builder(fixture),'additive_only',params,evidence)
 regressions={'balanced_fixture_PASS':base['status']=='PASS'}
 bad=[dict(r) for r in fixture];bad[-1]['tcr_cdr3_beta']='CASSD'
 ad,_=api.admit(bad,[1]*4,builder(bad),'additive_only',params,evidence);regressions['broken_TCR_denied']=ad['gates']['tcr_counts']=='FAIL'
 bad=[dict(r) for r in fixture];bad[-1]['hla']='HLA-C*01:01';bad[-1]['pmhc_id']=bad[-1]['epitope']+'|'+bad[-1]['hla']
 ad,_=api.admit(bad,[1]*4,builder(bad),'additive_only',params,evidence);regressions['broken_full_query_denied']=ad['gates']['complete_query_counts']=='FAIL'
 ad,_=api.admit(fixture,[2,1,1,1],builder(fixture),'additive_only',params,evidence);regressions['broken_weights_denied']=ad['gates']['effective_weights']=='FAIL'
 m=builder(fixture).tolil();m[2,0]+=.1
 ad,_=api.admit(fixture,[1]*4,m.tocsr(),'additive_only',params,evidence);regressions['pair_dependent_features_denied']=ad['gates']['separable_features']=='FAIL'
 for name,value in [('fit_intercept',False),('C',0),('class_weight','balanced')]:
  ad,_=api.admit(fixture,[1]*4,builder(fixture),'additive_only',{**params,name:value},evidence);regressions[name+'_unsupported_denied']=ad['gates']['actual_l2_and_intercept']=='FAIL'
 for name in ('identity','train_only'):
  ad,_=api.admit(fixture,[1]*4,builder(fixture),'additive_only',params,{**evidence,name:None});regressions[name+'_missing_UNVERIFIED']=ad['gates'][name]=='UNVERIFIED'
 assert all(regressions.values());write(w/'additive/rejection_regressions.json',{'status':'PASS','checks':regressions,'fit_calls':0})
 for directory in ('admissions','node_counts','models'):(w/'additive'/directory).mkdir(exist_ok=False)
 items=[];prepared=[]
 for t in tasks:
  graph=t['roles']['train'];rows=_fixed_rows(read(graph['positive']),read(graph['negative']),fold_id=t['fold_id'],partition='train')
  hla=fit_hla_mapping(rows);matrix=build_additive_feature_matrix(rows,hla);dimensions=additive_feature_dimensions(hla)
  stripped=[{k:r[k] for k in ('tcr_cdr3_beta','epitope','hla')} for r in rows]
  semantic={'label_metadata_invariance':(matrix!=build_additive_feature_matrix(stripped,hla)).nnz==0,'train_HLA_mapping':hla==fit_hla_mapping(stripped),'pure_marginal_dimension':matrix.shape==(len(rows),842+len(hla)) and dimensions['interaction']==0}
  assert sha(t['old_additive']['path'])==t['old_additive']['sha256']
  old=joblib.load(t['old_additive']['path']);semantic['R01_artifact_hla_and_params_correspond']=old['hla_mapping']==hla and old['estimator'].get_params(deep=False)==params and old['analytic_solution'] is True and old['optimizer_fit_performed'] is False
  ad,nodes=api.admit(rows,[1]*len(rows),matrix,'additive_only',params,{**evidence,'identity':all(semantic.values())})
  p=w/'additive/node_counts'/(t['task_id']+'.csv.gz')
  with p.open('xb') as raw:
   with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as z:
    with io.TextIOWrapper(z,encoding='utf-8',newline='') as f:
     writer=csv.DictWriter(f,fieldnames=list(nodes[0]),lineterminator='\n');writer.writeheader();writer.writerows(nodes)
  item={'task_id':t['task_id'],'admission':ad,'semantic_checks':semantic,'train_inputs':graph,'params':params,'hla_mapping':hla,'hla_mapping_sha256':digest(hla),'feature_dimensions':dimensions,'objective':'sum logloss + (||w||^2 + (b/intercept_scaling)^2)/(2C), liblinear penalized synthetic intercept','weights':'exact unit; no sample_weight; class_weight=None','no_prediction_near_half_admission':True,'node_counts_path':str(p),'node_counts_sha256':sha(p),'fit_calls':0}
  path=w/'additive/admissions'/(t['task_id']+'.json');write(path,item)
  items.append({'task_id':t['task_id'],'status':ad['status'],'all_semantic_checks':all(semantic.values()),'admission_path':str(path),'admission_sha256':sha(path),**ad['gates']});prepared.append((t,item,old))
 write(w/'additive/admission_index.json',{'count':len(items),'admissions':items})
 with (w/'additive/admission_table.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
 if not all(r['status']=='PASS' and r['all_semantic_checks'] for r in items):
  write(w/'additive/artifact_index.json',{'status':'BLOCKED_FAIL_OR_UNVERIFIED_NO_NEW_ANALYTIC_ARTIFACTS','artifacts':[],'fits':0});raise SystemExit(2)
 artifacts=[]
 for t,item,old in prepared:
  if t['interaction_fit']:
   path=w/'additive/models'/(t['task_id']+'.joblib')
   parent={'estimator':LogisticRegression(**params),'method':'additive_only','task_id':t['task_id'],'resample_id':t['train_resample_id'],'fold_id':t['fold_id'],'seed':13,'model_config':contract['model_config'],'feature_config':old['feature_config'],'feature_dimensions':item['feature_dimensions'],'hla_mapping':item['hla_mapping'],'feature_fit_partitions':['train'],'model_fit_partitions':['train'],'input_audit':t['roles']}
   api.analytic_artifact(path,item['admission'],parent,item['feature_dimensions']['total'],{'generation':'r05_original_repair_h1_20261009','parent_kind':'new_input_bound_task','task_id':t['task_id'],'admission_sha256':sha(w/'additive/admissions'/(t['task_id']+'.json')),'prior_R01_analytic_template':t['old_additive'],'new_train_inputs':t['roles']['train'],'R01_api_sha256':API_SHA,'R01_static_sha256':STATIC_SHA})
   action='NEW_INPUT_BOUND_ANALYTIC_ARTIFACT'
  else:path=Path(t['old_additive']['path']);action='VERIFIED_R01_BYTE_REUSE_NEW_TASK_WRAPPER'
  cold=joblib.load(path);est=cold['estimator'];checks={'coef_exact_zero':bool(np.all(est.coef_==0)),'intercept_exact_zero':bool(np.all(est.intercept_==0)),'zero_optimizer_iterations':bool(np.all(est.n_iter_==0)),'params_exact':est.get_params(deep=False)==params,'mapping_exact':cold['hla_mapping']==item['hla_mapping'],'admission_gates_PASS':all(v=='PASS' for v in cold['admission_gates'].values()),'fit_never_called':cold['optimizer_fit_performed'] is False}
  for role in ('train','validation','test'):
   g=t['roles'][role];rr=_fixed_rows(read(g['positive']),read(g['negative']),fold_id=t['fold_id'],partition=role);xx=build_additive_feature_matrix(rr,cold['hla_mapping'])
   checks[role+'_reload_exact_half']=bool(np.all(est.decision_function(xx)==0) and np.all(est.predict_proba(xx)==.5))
  novel=[{'tcr_cdr3_beta':'ACDEFGHIKLMNPQRSTVWY'*4,'epitope':'YWVTSRQPNMLKIHGFEDCA'*3,'hla':'HLA-H1*99:99'}];xx=build_additive_feature_matrix(novel,cold['hla_mapping'])
  checks['unknown_HLA_novel_exact_half']=bool(np.all(est.decision_function(xx)==0) and np.all(est.predict_proba(xx)==.5))
  artifacts.append({'task_id':t['task_id'],'action':action,'path':str(path),'sha256':sha(path),'status':'PASS' if all(checks.values()) else 'FAIL','reload_checks':checks,'hla_mapping_sha256':item['hla_mapping_sha256'],'latest_roles':t['roles'],'old_artifact':t['old_additive'],'admission_path':str(w/'additive/admissions'/(t['task_id']+'.json'))})
 passed=len(artifacts)==25 and sum(x['action']=='NEW_INPUT_BOUND_ANALYTIC_ARTIFACT' for x in artifacts)==18 and all(x['status']=='PASS' for x in artifacts)
 write(w/'additive/artifact_index.json',{'status':'PASS25_ANALYTIC_ADMISSIONS_AND_RELOADS' if passed else 'FAIL_RELOAD','count':25,'new':18,'reuse':7,'numerical_fits':0,'artifacts':artifacts})
 if not passed:raise SystemExit(2)
 print('PASS25_ANALYTIC_ADMISSIONS_AND_RELOADS',flush=True)
if __name__=='__main__':main()
