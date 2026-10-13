"""H1 original-family repair: 18 shared fits, no scoring and no additive fits."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,multiprocessing,platform,time,traceback,warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import joblib,numpy as np,scipy,sklearn
from sklearn.exceptions import ConvergenceWarning

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canon(x).encode()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def arrayhash(a):return hashlib.sha256(canon({'shape':a.shape,'dtype':str(a.dtype)}).encode()+a.tobytes()).hexdigest()
def matrixhash(m):
 h=hashlib.sha256(canon({'shape':m.shape,'dtype':str(m.dtype),'indices_dtype':str(m.indices.dtype),'indptr_dtype':str(m.indptr.dtype)}).encode())
 for a in (m.data,m.indices,m.indptr):h.update(a.tobytes())
 return h.hexdigest()
def read(entry):
 p=Path(entry['path']);assert sha(p)==entry['sha256'],str(p)
 op=gzip.open if p.suffix=='.gz' else open
 with op(p,'rt',newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 assert len(rows)==entry['rows']
 return rows

def fit(task,workstr,contract):
 work=Path(workstr);key=task['task_id'];folder=work/'training/tasks'/key;folder.mkdir(parents=True,exist_ok=False)
 result={'task_id':key,'action':'NEW_FIT','status':'FAIL','fit_calls':0,'model_path':'','model_sha256':'','score_calls':0}
 start=time.monotonic()
 try:
  from src.interaction_resample_robustness import _fixed_rows,_model_from_config
  from src.degree_matched_orthogonalized_logistic import _build_feature_matrix_and_audit,orthogonalized_feature_dimensions
  from src.kmer_logistic import fit_hla_mapping
  graph=task['roles']['train'];rows=_fixed_rows(read(graph['positive']),read(graph['negative']),fold_id=task['fold_id'],partition='train')
  hla=fit_hla_mapping(rows);template=joblib.load(task['old_model']['path'])
  assert sha(task['old_model']['path'])==task['old_model']['sha256']
  assert hla==template['hla_mapping'],'same endpoint degrees must preserve original training HLA mapping'
  X,audit=_build_feature_matrix_and_audit(rows,hla);dimensions=orthogonalized_feature_dimensions(hla)
  assert X.shape==(len(rows),1242+len(hla)) and np.isfinite(X.data).all()
  labels=np.asarray([r['label'] for r in rows],dtype=np.int64);assert np.array_equal(np.unique(labels),[0,1])
  model=_model_from_config(contract['model_config'],13);params=model.get_params(deep=False)
  assert params==contract['actual_estimator_params']==template['estimator'].get_params(deep=False)
  mh=matrixhash(X);lh=arrayhash(labels)
  write(folder/'fit_attempt.json',{'task_id':key,'registered_fit_calls':1,'utc_before_call':datetime.now(timezone.utc).isoformat(),'call':'model.fit(train_features,train_labels)','sample_weight_passed':False,'params':params,'train_matrix_sha256':mh,'train_labels_sha256':lh,'task_mapping_sha256':sha(work/'training/task_mapping25.json')})
  result['fit_calls']=1
  with warnings.catch_warnings(record=True) as captured:
   warnings.simplefilter('always');model.fit(X,labels)
  warn=[{'category':c.category.__name__,'message':str(c.message),'convergence_warning':issubclass(c.category,ConvergenceWarning)} for c in captured]
  its=[int(v) for v in model.n_iter_]
  assert not any(c['convergence_warning'] for c in warn) and max(its)<params['max_iter'],'failed convergence; no retry'
  assert matrixhash(X)==mh and np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()
  assert model.coef_.shape==(1,dimensions['total']) and np.array_equal(model.classes_,[0,1])
  additive_cols=np.r_[np.arange(840),np.arange(1240,dimensions['total'])]
  binding={role:{kind:task['roles'][role][kind] for kind in ('positive','negative')} for role in ('train','validation','test')}
  payload={'estimator':model,'method':'orthogonalized_interaction','resample_id':task['train_resample_id'],'fold_id':task['fold_id'],'seed':13,'model_config':contract['model_config'],'feature_config':template['feature_config'],'feature_dimensions':dimensions,'hla_mapping':hla,'feature_fit_partitions':['train'],'model_fit_partitions':['train'],'input_audit':binding,'interaction_orthogonality':{'train':audit},'generation':'r05_original_repair_h1_20261009','task_id':key,'model_id':'H1:orthogonalized_interaction:'+key,'original_checkpoint':task['old_model'],'training_task_sha256':digest(task),'pretraining_identity_sha256':sha(work/'training/pretraining_identity.json'),'optimizer_fit_performed':True,'sample_weight_passed':False,'score_calls_at_save':0}
  path=folder/'model.joblib'
  with path.open('xb') as f:joblib.dump(payload,f,compress=3)
  cold=joblib.load(path)
  assert np.array_equal(cold['estimator'].coef_,model.coef_) and np.array_equal(cold['estimator'].intercept_,model.intercept_) and cold['hla_mapping']==hla and cold['estimator'].get_params(deep=False)==params
  meta={'task_id':key,'status':'PASS','generation':payload['generation'],'model_path':str(path),'model_sha256':sha(path),'old_checkpoint':task['old_model'],'params':params,'model_config':contract['model_config'],'hla_mapping':hla,'hla_mapping_sha256':digest(hla),'feature_dimensions':dimensions,'feature_fit_partitions':['train'],'model_fit_partitions':['train'],'actual_fit_call':'model.fit(train_features,train_labels)','sample_weight_passed':False,'effective_weight_rule':'unit_no_sample_weight_class_weight_none','validation_selection_dependency':False,'orthogonalization':'original fixed double centering; no learned all-data transform','train_matrix_sha256':mh,'train_labels_sha256':lh,'train_orthogonality':audit,'input_audit':binding,'iterations':its,'converged':True,'warnings':warn,'coef_sha256':arrayhash(model.coef_),'intercept_sha256':arrayhash(model.intercept_),'joint_additive_block_l2_norm':float(np.linalg.norm(model.coef_[:,additive_cols])),'interaction_block_l2_norm':float(np.linalg.norm(model.coef_[:,840:1240])),'joint_additive_block_manually_modified':False,'serialization_parameter_roundtrip_exact':True,'score_calls':0,'elapsed_seconds':time.monotonic()-start}
  write(folder/'model_metadata.json',meta)
  result.update({'status':'PASS','model_path':str(path),'model_sha256':sha(path),'metadata_path':str(folder/'model_metadata.json'),'metadata_sha256':sha(folder/'model_metadata.json'),'iterations':its,'warnings':warn,'elapsed_seconds':time.monotonic()-start})
 except Exception as e:
  result.update({'exception':type(e).__name__+': '+str(e),'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-start});write(folder/'failure.json',result)
 write(folder/'task_result.json',result)
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);a=ap.parse_args();w=a.work
 identity=load(w/'training/pretraining_identity.json');assert identity['status']=='FROZEN_BEFORE_FITS_AND_SCORES'
 for item in identity['files']:assert sha(w/item['path'])==item['sha256']
 mapping=load(w/'training/task_mapping25.json');tasks=mapping['tasks'];contract=load(w/'authority/model_contract.json')
 assert len(tasks)==25 and sum(t['interaction_fit'] for t in tasks)==18
 assert sklearn.__version__=='1.7.2' and mapping['interaction_fit_budget']==18 and mapping['additive_numerical_fit_budget']==0
 assert load(w/'additive/artifact_index.json')['status']=='PASS25_ANALYTIC_ADMISSIONS_AND_RELOADS'
 assert load(w/'training/entry_gate.json')['status']=='PASS_STRUCTURE_AND_REUSE_DEPENDENCIES'
 assert not (w/'training/tasks').exists(),'no duplicate fit execution'
 (w/'training/tasks').mkdir()
 write(w/'training/runtime_identity.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__,'docker_image':'sha256:d7c8838e67a156a9ed1abe981fee56f55a3637f2af47bae7b256a3c3c7551b59','workers':6,'blas_threads':1,'start_method':'spawn','interaction_fit_budget':18,'additive_numeric_fits':0,'extra_external_fits':0,'score_calls_in_training':0})
 results=[]
 with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
  futures=[pool.submit(fit,t,str(w),contract) for t in tasks if t['interaction_fit']]
  for future in as_completed(futures):
   result=future.result();results.append(result);print(canon({'task_id':result['task_id'],'status':result['status'],'completed':len(results),'fit_calls':sum(r['fit_calls'] for r in results)}),flush=True)
 results.sort(key=lambda r:r['task_id']);assert sum(r['fit_calls'] for r in results)<=18
 write(w/'training/fit_ledger.json',results);write(w/'training/failures.json',[r for r in results if r['status']!='PASS'])
 passing=len(results)==18 and all(r['status']=='PASS' and r['fit_calls']==1 for r in results)
 write(w/'training/fit_summary.json',{'status':'PASS18_SHARED_FITS' if passing else 'FAIL_STOP_NO_RETRY','required_fits':18,'completed':len(results),'fit_calls':sum(r['fit_calls'] for r in results),'passing':sum(r['status']=='PASS' for r in results),'additive_numeric_fits':0,'extra_external_fits':0,'scores_in_training':0})
 if not passing:raise SystemExit(2)
 bykey={r['task_id']:r for r in results};index=[]
 for task in tasks:
  r=bykey.get(task['task_id']);model={'path':r['model_path'],'sha256':r['model_sha256']} if r else task['old_model']
  assert sha(model['path'])==model['sha256']
  index.append({'task_id':task['task_id'],'train_resample_id':task['train_resample_id'],'fold_id':task['fold_id'],'action':'NEW_FIT' if r else 'VERIFIED_REUSE','model_path':model['path'],'model_sha256':model['sha256'],'old_model':task['old_model'],'roles':task['roles'],'additive_path':task['additive']['path'],'additive_sha256':task['additive']['sha256'],'additive_action':task['additive']['action'],'new_task_binding_wrapper':True,'original_training_provenance_preserved':not bool(r),'status':'PASS'})
 write(w/'training/model_index25.json',{'status':'PASS25_SHARED_INTERNAL_EXTERNAL_MEMBERS','count':25,'new_fits':18,'reused':7,'members':index})
if __name__=='__main__':main()
