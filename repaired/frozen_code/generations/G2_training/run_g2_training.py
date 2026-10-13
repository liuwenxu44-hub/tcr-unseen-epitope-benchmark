"""Exactly one original interaction Logistic fit per frozen G2 task."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,io,json,multiprocessing,platform,time,traceback,warnings
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import joblib,numpy as np,scipy,sklearn
from sklearn.exceptions import ConvergenceWarning

G2_SHA='98bfca307c68f01f6980f3f2f0d6bf01529254cd554ffff4a253d6fc54942d35'
ROLES=('train','validation','test')
PRED_FIELDS=('recipe_id','fold_id','draw_id','partition','model_name','model_id','model_sha256','input_source_file','input_source_sha256','input_source_row_index','tcr_cdr3_beta','epitope','hla','pmhc_id','label','probability','logit')
def cj(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True)
def digest(x):return hashlib.sha256(cj(x).encode('ascii')).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text())
def writej(p,value):
 with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def matrixsha(m):
 h=hashlib.sha256(cj({'shape':m.shape,'dtype':str(m.dtype),'indices_dtype':str(m.indices.dtype),'indptr_dtype':str(m.indptr.dtype)}).encode('ascii'))
 for x in (m.data,m.indices,m.indptr):h.update(x.tobytes())
 return h.hexdigest()
def arrayhash(a):return hashlib.sha256(cj({'shape':a.shape,'dtype':str(a.dtype)}).encode('ascii')+a.tobytes()).hexdigest()

def read_role(task,role,g2,manifest):
 rows=[]
 for kind,label in (('positive',1),('negative',0)):
  entry=task['roles'][role][kind];relative=entry['relative_path'];p=g2/relative
  assert p.resolve().is_relative_to(g2.resolve()) and relative in manifest
  assert sha(p)==entry['sha256']==manifest[relative]['sha256'] and p.stat().st_size==entry['bytes']==manifest[relative]['bytes']
  with gzip.open(p,'rt',encoding='utf-8',newline='') as f:original=list(csv.DictReader(f))
  assert len(original)==entry['rows']
  for i,r in enumerate(original):
   assert int(r['label'])==label
   assert tuple(r[k] for k in ('tcr_cdr3_beta','epitope','hla'))==tuple(r[k].strip().upper() for k in ('tcr_cdr3_beta','epitope','hla'))
   assert r['pmhc_id']==r['epitope']+'|'+r['hla']
   if kind=='negative':
    assert r['recipe_id']==task['recipe_id'] and r['fold_id']==task['fold_id'] and r['partition']==role and r['draw_id']==task['draw_id']
    assert int(r['derived_seed'])==task['source_inputs'][role]['negative_derived_seed'] and int(r['base_seed'])==task['negative_draw_base_seed']
   rows.append({**r,'label':label,'input_source_file':relative,'input_source_sha256':entry['sha256'],'input_source_row_index':i})
 return rows

def worker(task,workstr,g2str,sourcestr,contract,manifest):
 work,g2,source=map(Path,(workstr,g2str,sourcestr));r,f,d=(task[k] for k in ('recipe_id','fold_id','draw_id'))
 folder=work/'execution/tasks'/r/f/d;folder.mkdir(parents=True,exist_ok=False)
 model_id=f'R04G2:{r}:{f}:{d}:orthogonalized_interaction'
 result={'task_id':task['task_id'],'recipe_id':r,'fold_id':f,'draw_id':d,'status':'FAIL','fit_attempts':0,'model_id':model_id,'model_name':'orthogonalized_interaction','model_file':'','model_sha256':'','prediction_file':'','prediction_sha256':'','metadata_file':'','source_inputs':task['source_inputs'],'complete_input_task_sha256':task['complete_input_task_sha256'],'model_seed':13,'checkpoint_reuse':False}
 started=time.monotonic();warning_records=[]
 try:
  from src.interaction_resample_robustness import _model_from_config
  from src.degree_matched_orthogonalized_logistic import _build_feature_matrix_and_audit,orthogonalized_feature_dimensions
  from src.kmer_logistic import fit_hla_mapping
  rows={role:read_role(task,role,g2,manifest) for role in ROLES}
  hla_mapping=fit_hla_mapping(rows['train']);matrices={};audits={}
  for role in ROLES:
   m,audit=_build_feature_matrix_and_audit(rows[role],hla_mapping)
   assert np.isfinite(m.data).all() and m.shape[0]==len(rows[role])
   matrices[role]=m;audits[role]={'matrix_sha256':matrixsha(m),'shape':list(m.shape),'orthogonality':audit,'hla_mapping_sha256':digest(hla_mapping),'feature_fit_partitions':['train']}
  dimensions=orthogonalized_feature_dimensions(hla_mapping)
  assert dimensions['interaction']==400 and matrices['train'].shape[1]==1242+len(hla_mapping)
  model=_model_from_config(contract['original_model_config'],13)
  params=model.get_params(deep=False);assert params==contract['actual_estimator_params']
  labels=np.asarray([row['label'] for row in rows['train']],dtype=np.int64)
  assert np.array_equal(np.unique(labels),np.array([0,1]))
  writej(folder/'fit_attempt.json',{'task_id':task['task_id'],'fit_invocations_registered':1,'utc_before_call':datetime.now(timezone.utc).isoformat(),'fit_call':'model.fit(train_features, train_labels)','params':params,'input_task_sha256':task['complete_input_task_sha256'],'train_matrix_sha256':audits['train']['matrix_sha256'],'train_labels_sha256':arrayhash(labels),'sample_weight_passed':False})
  result['fit_attempts']=1
  with warnings.catch_warnings(record=True) as captured:
   warnings.simplefilter('always')
   model.fit(matrices['train'],labels)
  warning_records=[{'category':type(x.message).__name__,'message':str(x.message),'convergence_warning':issubclass(x.category,ConvergenceWarning)} for x in captured]
  iterations=[int(x) for x in model.n_iter_]
  result['iterations']=iterations;result['warnings']=warning_records
  assert matrixsha(matrices['train'])==audits['train']['matrix_sha256'],'training matrix mutated during fit'
  assert np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()
  assert not any(x['convergence_warning'] for x in warning_records) and max(iterations)<params['max_iter'],'convergence acceptance failed; no retry'
  assert model.coef_.shape==(1,dimensions['total']) and np.array_equal(model.classes_,np.array([0,1]))
  logits=np.asarray(model.decision_function(matrices['test']),dtype=np.float64);probabilities=np.asarray(model.predict_proba(matrices['test'])[:,1],dtype=np.float64)
  assert np.isfinite(logits).all() and np.isfinite(probabilities).all() and np.all((probabilities>=0)&(probabilities<=1))
  model_path=folder/'model.joblib';prediction_path=folder/'test_predictions.csv.gz';metadata_path=folder/'model_metadata.json'
  relative=lambda p:p.relative_to(work).as_posix()
  checkpoint={'estimator':model,'model_id':model_id,'method':'orthogonalized_interaction','task_id':task['task_id'],'recipe_id':r,'fold_id':f,'draw_id':d,'seed':13,'hla_mapping':hla_mapping,'feature_dimensions':dimensions,'model_config':contract['original_model_config'],'params':params,'feature_fit_partitions':['train'],'model_fit_partitions':['train'],'source_inputs':task['source_inputs'],'complete_input_task_sha256':task['complete_input_task_sha256'],'interaction_transform':'double_centered_20x20','training_code_sha256':sha(work/'run_g2_training.py')}
  with model_path.open('xb') as stream:joblib.dump(checkpoint,stream,compress=3)
  model_sha=sha(model_path)
  reloaded=joblib.load(model_path);rp=reloaded['estimator'].predict_proba(matrices['test'])[:,1];rl=reloaded['estimator'].decision_function(matrices['test'])
  assert np.array_equal(probabilities,rp) and np.array_equal(logits,rl),'checkpoint roundtrip changed predictions'
  with prediction_path.open('xb') as raw:
   with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as z:
    with io.TextIOWrapper(z,encoding='utf-8',newline='') as stream:
     writer=csv.DictWriter(stream,fieldnames=PRED_FIELDS,lineterminator='\n');writer.writeheader()
     for row,l,p in zip(rows['test'],logits,probabilities,strict=True):
      writer.writerow({**{k:row[k] for k in ('input_source_file','input_source_sha256','input_source_row_index','tcr_cdr3_beta','epitope','hla','pmhc_id','label')},'recipe_id':r,'fold_id':f,'draw_id':d,'partition':'test','model_name':'orthogonalized_interaction','model_id':model_id,'model_sha256':model_sha,'logit':float(l),'probability':float(p)})
  coefficient=model.coef_.copy();additive_columns=np.r_[np.arange(840),np.arange(1240,dimensions['total'])]
  meta={'task_id':task['task_id'],'recipe_id':r,'fold_id':f,'draw_id':d,'status':'PASS','model_id':model_id,'model_name':'orthogonalized_interaction','model_file':relative(model_path),'model_sha256':model_sha,'params':params,'source_inputs':task['source_inputs'],'complete_input_task_sha256':task['complete_input_task_sha256'],'hla_mapping':hla_mapping,'hla_mapping_sha256':digest(hla_mapping),'feature_dimensions':dimensions,'feature_audits':audits,'feature_fit_partitions':['train'],'model_fit_partitions':['train'],'sample_weight_passed':False,'effective_weight_rule':'unit_no_sample_weight_class_weight_none','codeidentities':contract['codeidentities'],'fit_attempts':1,'iterations':iterations,'converged':True,'warnings':warning_records,'coef_sha256':arrayhash(coefficient),'intercept_sha256':arrayhash(model.intercept_),'jointly_fitted_additive_block_l2_norm':float(np.linalg.norm(coefficient[:,additive_columns])),'interaction_block_l2_norm':float(np.linalg.norm(coefficient[:,840:1240])),'joint_additive_block_manually_modified':False,'serialized_predictions_exactly_identical':True,'validation_role':'train-map feature transform/audit only; no model fitting, selection or early stopping','probability_implementation':'estimator.predict_proba(test_features)[:,1]; no rounding/clipping/jitter','training_code_sha256':sha(work/'run_g2_training.py'),'pretraining_identity_sha256':sha(work/'pretraining_identity.json'),'elapsed_seconds':time.monotonic()-started}
  writej(metadata_path,meta)
  result.update({'status':'PASS','model_file':relative(model_path),'model_sha256':model_sha,'prediction_file':relative(prediction_path),'prediction_sha256':sha(prediction_path),'metadata_file':relative(metadata_path),'metadata_sha256':sha(metadata_path),'params':params,'iterations':iterations,'warnings':warning_records,'test_rows':len(rows['test']),'elapsed_seconds':time.monotonic()-started})
 except Exception as error:
  result.update({'exception_type':type(error).__name__,'exception':str(error),'traceback':traceback.format_exc(),'warnings':warning_records,'elapsed_seconds':time.monotonic()-started})
  writej(folder/'failure.json',result)
 writej(folder/'task_result.json',result)
 return result

def main():
 ap=argparse.ArgumentParser()
 for k in ('work','g2','source'):ap.add_argument('--'+k,type=Path,required=True)
 a=ap.parse_args();w,g2,source=a.work,a.g2,a.source
 assert sha(g2/'g2_manifest.json')==G2_SHA
 identity=load(w/'pretraining_identity.json')
 assert identity['status']=='FROZEN_BEFORE_NUMERICAL_FITS'
 for x in identity['files']:assert sha(w/x['path'])==x['sha256'] and (w/x['path']).stat().st_size==x['bytes'],x['path']
 contract=load(w/'authority/model_contract.json');mapping=load(w/'authority/task_mapping105.json');scope=load(w/'training_protocol.json')
 assert mapping['model_contract_sha256']==sha(w/'authority/model_contract.json') and mapping['task_count']==mapping['interaction_fit_max']==105
 assert sklearn.__version__=='1.7.2' and scope['workers']==8 and scope['multiprocessing_start_method']=='spawn'
 for x in contract['codeidentities']:assert sha(source/x['relative_path'])==x['sha256'],x['relative_path']
 tasks=mapping['tasks'];assert len(tasks)==len({t['task_id'] for t in tasks})==105 and all(t['model_seed']==13 and not t['checkpoint_reuse'] for t in tasks)
 additive=load(w/'additive/artifact_index.json')['artifacts']
 assert len(additive)==105 and {t['task_id'] for t in additive}=={t['task_id'] for t in tasks} and all(t['status']=='PASS' for t in additive),'New task analytic admission is incomplete'
 manifest={x['path']:x for x in load(g2/'g2_manifest.json')['files']}
 assert not any((w/'execution').iterdir()),'Refusing duplicate numerical fitting or overwriting an execution'
 writej(w/'execution/runtime_identity.json',{'utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__,'g2_manifest_sha256':G2_SHA,'pretraining_identity_sha256':sha(w/'pretraining_identity.json'),'workers':8,'blas_threads':1,'multiprocessing_start_method':'spawn','interaction_fit_budget':105,'additive_numerical_fits':0,'CNN_fits':0,'external_scores':0})
 results=[]
 with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as executor:
  futures={executor.submit(worker,t,str(w),str(g2),str(source),contract,manifest):t for t in tasks}
  for future in as_completed(futures):
   t=futures[future]
   try:result=future.result()
   except Exception as e:result={'task_id':t['task_id'],**{k:t[k] for k in ('recipe_id','fold_id','draw_id')},'status':'FAIL','fit_attempts':int((w/'execution/tasks'/t['recipe_id']/t['fold_id']/t['draw_id']/'fit_attempt.json').exists()),'exception_type':type(e).__name__,'exception':str(e),'traceback':traceback.format_exc()}
   results.append(result)
   if len(results)%5==0:print(cj({'completed':len(results),'passing':sum(x['status']=='PASS' for x in results),'failed':sum(x['status']!='PASS' for x in results),'required':105}),flush=True)
 results.sort(key=lambda x:(x['recipe_id'],x['fold_id'],x['draw_id']))
 assert sum(r['fit_attempts'] for r in results)<=105
 writej(w/'execution/fit_ledger.json',results);writej(w/'execution/failure_records.json',[r for r in results if r['status']!='PASS'])
 summary={'status':'PASS_ALL105_INTERACTION_FITS' if all(r['status']=='PASS' for r in results) else 'COMPLETE_WITH_FAILED_TASKS_NO_RETRY','required_tasks':105,'completed_tasks':len(results),'interaction_fit_calls':sum(r['fit_attempts'] for r in results),'passing_tasks':sum(r['status']=='PASS' for r in results),'failed_tasks':sum(r['status']!='PASS' for r in results),'additive_numerical_fits':0,'CNN_fits':0,'external_scores':0,'hyperparameters_seeds_inputs_changed':False,'manuscript_figures_changed':False,'inferential_statistics_computed':False,'finished_utc':datetime.now(timezone.utc).isoformat()}
 writej(w/'execution/execution_summary.json',summary)
 files=[]
 # Only fixed producer-owned paths; no source selection by recursive walk.
 for name in ('runtime_identity.json','fit_ledger.json','failure_records.json','execution_summary.json'):files.append('execution/'+name)
 for t in tasks:
  base='execution/tasks/'+t['recipe_id']+'/'+t['fold_id']+'/'+t['draw_id']+'/'
  for name in ('fit_attempt.json','model.joblib','test_predictions.csv.gz','model_metadata.json','failure.json','task_result.json'):
   if (w/(base+name)).is_file():files.append(base+name)
 writej(w/'execution/execution_output_catalog.json',{'files':[{'path':p,'bytes':(w/p).stat().st_size,'sha256':sha(w/p),'role':'INTERACTION_MODEL_PREDICTION_OR_FIT_AUDIT'} for p in files]})
 print(cj(summary),flush=True)
if __name__=='__main__':main()
