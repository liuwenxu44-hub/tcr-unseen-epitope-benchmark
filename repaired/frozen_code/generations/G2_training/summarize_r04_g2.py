#!/usr/bin/env python3
"""Frozen fixed-grid R04 G2 descriptive metrics. No training or inference."""
from __future__ import annotations
import argparse, csv, gzip, hashlib, io, json, math, statistics, traceback
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

RECIPES = tuple('S'+str(i) for i in range(7))
FOLDS = tuple('fold_'+str(i) for i in range(1,6))
DRAWS = tuple('draw_'+str(i) for i in range(1,4))
G2_SHA = '98bfca307c68f01f6980f3f2f0d6bf01529254cd554ffff4a253d6fc54942d35'
PRED_COLS = ['recipe_id','fold_id','draw_id','partition','model_name','model_id','model_sha256','input_source_file','input_source_sha256','input_source_row_index','tcr_cdr3_beta','epitope','hla','pmhc_id','label','probability','logit']
BASE_METRICS = ['interaction_AP','interaction_AUROC','additive_AP','additive_AUROC','finite_no_ties_expected_AP','finite_no_ties_expected_AUROC','delta_Additive_AP','delta_Additive_AUROC','delta_finite_no_ties_AP','delta_finite_no_ties_AUROC']
Q_COLS = ['recipe_id','draw_id','fold_id','epitope','hla','pmhc_id','npositive','nnegative','sample_count','unique_score_count','tied_group_count','records_in_ties','max_tie_size',*BASE_METRICS,'status']
DELTA_COLS = ['delta_G2_S0_'+x for x in BASE_METRICS]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def csv_read(p):
    op=gzip.open if str(p).endswith('.gz') else open
    with op(p,'rt',encoding='utf-8',newline='') as f:
        r=csv.DictReader(f); cols=r.fieldnames; rows=list(r)
    return cols,rows

def csv_write(p,cols,rows):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): raise FileExistsError('Refuse overwrite: '+str(p))
    s=io.StringIO(newline=''); w=csv.DictWriter(s,fieldnames=cols,lineterminator='\n'); w.writeheader()
    for r in rows: w.writerow({c:r.get(c,'') for c in cols})
    data=s.getvalue().encode()
    p.write_bytes(gzip.compress(data,mtime=0) if str(p).endswith('.gz') else data)

def json_write(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): raise FileExistsError('Refuse overwrite: '+str(p))
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def within(root,rel):
    p=(root/rel).resolve(); p.relative_to(root.resolve()); return p

def key(r): return (r['tcr_cdr3_beta'],r['epitope'],r['hla'],int(r['label']))
def query(r): return (r['epitope'],r['hla'])
def finite_ap(p,n):
    if not 0<p<n: raise ValueError('Finite ranking reference requires both classes')
    h=math.fsum(1/k for k in range(1,n+1))
    return h/n+(p-1)*(n-h)/(n*(n-1))
def means(rows,cols): return {c:math.fsum(float(r[c]) for r in rows)/len(rows) for c in cols}
def describe(rows,cols):
    out={}
    for c in cols:
        vals=[float(r[c]) for r in rows]; mean=math.fsum(vals)/len(vals)
        out.update({c+'_mean':mean,c+'_sd_ddof0':math.sqrt(math.fsum((v-mean)**2 for v in vals)/len(vals)),c+'_min':min(vals),c+'_max':max(vals),c+'_range':max(vals)-min(vals)})
    return out

def task_specs(g2):
    _,rows=csv_read(g2/'execution/input_results.csv')
    out={}
    for r in rows:
        k=(r['recipe_id'],r['fold_id'],r['partition'],r['draw_id'])
        if k in out: raise ValueError('Duplicate G2 task partition')
        out[k]=r
    expected={(s,f,p,d) for s in RECIPES for f in FOLDS for p in ('train','validation','test') for d in DRAWS}
    if set(out)!=expected: raise ValueError('G2 input catalog is not exact 315 partitions')
    return out

def run(root,g2):
    metrics=root/'metrics'; metrics.mkdir(exist_ok=True)
    outputs=[]; inputs={}; errors=[]
    def consume(p,expected=None,role='INPUT'):
        p=Path(p); digest=sha(p)
        if expected is not None and digest!=expected: raise ValueError('SHA mismatch: '+str(p))
        identifier=str(p.resolve())
        rec={'path':identifier,'bytes':p.stat().st_size,'sha256':digest,'role':role}
        if identifier in inputs and inputs[identifier]['sha256']!=digest: raise ValueError('Input drift during run')
        inputs[identifier]=rec;return digest
    def emit(rel,cols,rows):
        p=root/rel;csv_write(p,cols,rows);outputs.append({'role':'DESCRIPTIVE_METRIC_OUTPUT','path':rel,'bytes':p.stat().st_size,'sha256':sha(p)})
    consume(g2/'g2_manifest.json',G2_SHA,'G2_MANIFEST')
    manifest=json.loads((g2/'g2_manifest.json').read_text()); mapping={x['path']:x['sha256'] for x in manifest['files']}
    contract=json.loads((metrics/'calculation_contract.json').read_text()); consume(metrics/'calculation_contract.json',role='METRIC_CONTRACT')
    consume(root/'authority/model_contract.json',role='FROZEN_MODEL_CONTRACT'); model_contract=json.loads((root/'authority/model_contract.json').read_text()); expected_params=model_contract['actual_estimator_params']
    frozen=json.loads((metrics/'code_contract_manifest.json').read_text())
    for r in frozen['files']: consume(root/r['path'],r['sha256'],'FROZEN_METRIC_CODE_OR_CONTRACT')
    if contract['g2_manifest_sha256']!=G2_SHA or contract['descriptive_variation']['sd_ddof']!=0: raise ValueError('Metric contract changed')
    def g2_rows(rel):
        if rel not in mapping: raise ValueError('Unmanifested G2 input '+rel)
        consume(g2/rel,mapping[rel],'G2_INPUT');return csv_read(g2/rel)[1]
    common=g2_rows('execution/common_positives.csv');common_keys=Counter(map(key,common));common_q=Counter(query(r) for r in common)
    if any(v!=1 for v in common_keys.values()):raise ValueError('Common positive duplicate key')
    consume(g2/'execution/input_results.csv',mapping['execution/input_results.csv'],'G2_TASK_CATALOG');specs=task_specs(g2)
    ledger_path=root/'execution/fit_ledger.json'; consume(ledger_path,role='MODEL_FIT_LEDGER')
    ledger=json.loads(ledger_path.read_text());ledger=ledger['tasks'] if isinstance(ledger,dict) else ledger
    index_path=root/'additive/artifact_index.json';consume(index_path,role='ADDITIVE_ARTIFACT_INDEX')
    additives=json.loads(index_path.read_text());additives=additives['artifacts'] if isinstance(additives,dict) else additives
    def index(records):
        o={}
        for r in records:
            k=(r['recipe_id'],r['fold_id'],r['draw_id'])
            if k in o: raise ValueError('Duplicate task index key '+str(k))
            o[k]=r
        extra=set(o)-{(s,f,d) for s in RECIPES for f in FOLDS for d in DRAWS}
        if extra:raise ValueError('Unexpected task keys '+str(extra))
        return o
    li=index(ledger);ai=index(additives)
    paired={};audits=[]
    for s in RECIPES:
      for d in DRAWS:
       for f in FOLDS:
        tid=f'{s}__{f}__{d}';k=(s,f,d)
        ar={'task_id':tid,'recipe_id':s,'fold_id':f,'draw_id':d,'status':'FAIL','failure':''}
        try:
            l=li[k];a=ai[k]
            if l['status']!='PASS' or a['status']!='PASS':raise ValueError('Task or analytic artifact status not PASS')
            for record in (l,a):
                if record['task_id']!=tid:raise ValueError('task_id mismatch')
            if int(l['fit_attempts'])!=1:raise ValueError('fit_attempts must be fixed one; release cannot conceal retries')
            model_digest=consume(within(root,l['model_file']),l['model_sha256'],'INTERACTION_MODEL')
            consume(within(root,l['metadata_file']),role='MODEL_METADATA');meta=json.loads(within(root,l['metadata_file']).read_text())
            identity={'task_id':tid,'recipe_id':s,'fold_id':f,'draw_id':d,'model_id':f'R04G2:{s}:{f}:{d}:orthogonalized_interaction','model_name':'orthogonalized_interaction','model_file':l['model_file'],'model_sha256':model_digest}
            if any(meta.get(n)!=v for n,v in identity.items()):raise ValueError('Model metadata identity mismatch')
            if any(meta['params'].get(n)!=v or l['params'].get(n)!=v for n,v in expected_params.items()):raise ValueError('Frozen estimator parameter mismatch')
            if meta['model_fit_partitions']!=['train'] or meta['feature_fit_partitions']!=['train']:raise ValueError('Training/feature fit partition mismatch')
            if len(meta['hla_mapping_sha256'])!=64:raise ValueError('HLA mapping SHA missing')
            for p in ('train','validation','test'):
                spec=specs[(s,f,p,d)]; mi=meta['source_inputs'][p]
                for kind in ('positive','negative'):
                    rel=spec[kind+'_file'];expect={'file':rel,'sha256':mapping[rel],'rows':int(spec[kind+'_count'])}
                    if any(mi.get(kind+'_'+n)!=v for n,v in expect.items()):raise ValueError('Model source '+p+'/'+kind+' identity mismatch')
                if str(mi['negative_derived_seed'])!=str(spec['derived_seed']):raise ValueError('Negative derived seed mismatch')
            if not a.get('reload_checks') or not all(v is True for v in a['reload_checks'].values()):raise ValueError('Analytic artifact reload checks not all strict PASS')
            for part in ('train','validation','test'):
                ac=a['partition_checks'][part];sp=specs[(s,f,part,d)]
                if int(ac['rows'])!=int(sp['positive_count'])+int(sp['negative_count']) or any(ac.get(n) is not True for n in ('logit_exact_zero','probability_exact_half','feature_dimension_matches')):raise ValueError('Analytic artifact partition scoring evidence mismatch')
            adigest=consume(within(root,a['artifact_file']),a['artifact_sha256'],'ANALYTIC_ADDITIVE_ARTIFACT')
            predpath=within(root,l['prediction_file']);consume(predpath,l['prediction_sha256'],'TEST_PREDICTIONS');cols,pr=csv_read(predpath)
            if cols!=PRED_COLS:raise ValueError('Prediction column schema mismatch')
            spec=specs[(s,f,'test',d)];out=[];offset=0
            for kind in ('positive','negative'):
                rel=spec[kind+'_file'];src=g2_rows(rel)
                for idx,r in enumerate(src):
                    if offset>=len(pr):raise ValueError('Prediction rows missing')
                    p=pr[offset];offset+=1
                    expected={'recipe_id':s,'fold_id':f,'draw_id':d,'partition':'test','model_name':'orthogonalized_interaction','model_id':identity['model_id'],'model_sha256':model_digest,'input_source_file':rel,'input_source_sha256':mapping[rel],'input_source_row_index':str(idx)}
                    if any(p[n]!=v for n,v in expected.items()):raise ValueError('Prediction task or source row identity mismatch')
                    if key(p)!=key(r) or p['pmhc_id']!=r['pmhc_id']:raise ValueError('Prediction input key/label/query mismatch')
                    prob=float(p['probability']);logit=float(p['logit'])
                    if not math.isfinite(prob) or not 0<=prob<=1 or not math.isfinite(logit):raise ValueError('Nonfinite or out-of-bounds score')
                    logistic=1/(1+math.exp(-logit)) if logit>=0 else math.exp(logit)/(1+math.exp(logit))
                    if abs(prob-logistic)>1e-12:raise ValueError('Probability/logit mismatch')
                    out.append({**p,'additive_model_id':a['model_id'],'additive_artifact_file':a['artifact_file'],'additive_artifact_sha256':adigest,'analytic_additive_probability':0.5,'analytic_additive_logit':0.0})
            if offset!=len(pr):raise ValueError('Extra prediction rows')
            paired[k]=out;ar.update(status='PASS',row_count=len(out),model_sha256=model_digest,prediction_sha256=l['prediction_sha256'],additive_artifact_sha256=adigest)
        except Exception as exc:
            ar['failure']=type(exc).__name__+': '+str(exc);errors.append({'task_id':tid,'error':ar['failure']})
        audits.append(ar)
    emit('metrics/task_audit.csv',['task_id','recipe_id','fold_id','draw_id','status','row_count','model_sha256','prediction_sha256','additive_artifact_sha256','failure'],audits)
    expected_qfold={}
    for s in RECIPES:
      for f in FOLDS:
        for r in g2_rows(specs[(s,f,'test','draw_1')]['positive_file']):
            qq=query(r)
            if (s,qq) in expected_qfold and expected_qfold[s,qq]!=f:raise ValueError('Frozen query occurs in multiple test folds')
            expected_qfold[s,qq]=f
    perquery=[];drawrows=[];coverage=[]
    oof_cols=PRED_COLS+['additive_model_id','additive_artifact_file','additive_artifact_sha256','analytic_additive_probability','analytic_additive_logit']
    for s in RECIPES:
      for d in DRAWS:
        rows=[r for f in FOLDS for r in paired.get((s,f,d),[])];found=[f for f in FOLDS if (s,f,d) in paired]
        grouped=defaultdict(list)
        for r in rows:grouped[query(r)].append(r)
        actualpos=Counter(key(r) for r in rows if int(r['label'])==1)
        cq={'recipe_id':s,'draw_id':d,'complete_test_folds':len(found),'expected_test_folds':5,'expected_positive_count':len(common),'observed_positive_count':sum(actualpos.values()),'expected_query_count':len(common_q),'observed_query_count':len(grouped),'row_count':len(rows),'common_positive_coverage_exact':actualpos==common_keys,'common_query_coverage_exact':set(grouped)==set(common_q),'queries_in_multiple_test_folds':sum(len({r['fold_id'] for r in rr})!=1 for rr in grouped.values()),'balanced_query_count':sum(sum(int(r['label']) for r in rr)*2==len(rr) for rr in grouped.values()),'failure':''}
        complete=len(found)==5 and cq['common_positive_coverage_exact'] and cq['common_query_coverage_exact'] and cq['queries_in_multiple_test_folds']==0
        if not complete:cq['failure']='Missing/failed task or common-positive/query/fold coverage mismatch'
        local=[]
        for q in sorted(set(common_q)|set(grouped)):
            rr=grouped.get(q,[])
            y=np.asarray([int(r['label']) for r in rr]);score=np.asarray([float(r['probability']) for r in rr]);np_=int(y.sum());nn=len(y)-np_
            if np_<=0 or nn<=0:
                complete=False;cq['failure']='Missing or non-evaluable query retained'
                counts=Counter(score.tolist());tied=[v for v in counts.values() if v>=2]
                local.append({'recipe_id':s,'draw_id':d,'fold_id':expected_qfold.get((s,q),''),'epitope':q[0],'hla':q[1],'pmhc_id':q[0]+'|'+q[1],'npositive':np_,'nnegative':nn,'sample_count':len(y),'unique_score_count':len(counts),'tied_group_count':len(tied),'records_in_ties':sum(tied),'max_tie_size':max(counts.values(),default=0),**{m:'' for m in BASE_METRICS}})
                continue
            counts=Counter(score.tolist());tied=[v for v in counts.values() if v>=2]
            ap=float(average_precision_score(y,score));auc=float(roc_auc_score(y,score));add=np_/len(y);ref=finite_ap(np_,len(y))
            r={'recipe_id':s,'draw_id':d,'fold_id':rr[0]['fold_id'],'epitope':q[0],'hla':q[1],'pmhc_id':q[0]+'|'+q[1],'npositive':np_,'nnegative':nn,'sample_count':len(y),'unique_score_count':len(counts),'tied_group_count':len(tied),'records_in_ties':sum(tied),'max_tie_size':max(counts.values()),'interaction_AP':ap,'interaction_AUROC':auc,'additive_AP':add,'additive_AUROC':0.5,'finite_no_ties_expected_AP':ref,'finite_no_ties_expected_AUROC':0.5,'delta_Additive_AP':ap-add,'delta_Additive_AUROC':auc-0.5,'delta_finite_no_ties_AP':ap-ref,'delta_finite_no_ties_AUROC':auc-0.5}
            local.append(r)
        status='COMPLETE_DESCRIPTIVE' if complete else 'INCOMPLETE'
        for r in local:r['status']=status
        perquery.extend(local);cq['status']=status;coverage.append(cq)
        dr={'recipe_id':s,'draw_id':d,'query_count':len(grouped),'row_count':len(rows),'npositive':sum(int(r['label']) for r in rows),'nnegative':sum(1-int(r['label']) for r in rows),'status':status}
        if complete:dr.update(means(local,BASE_METRICS))
        drawrows.append(dr)
        # Preserve partial original rows, explicitly marked INCOMPLETE in coverage.
        emit(f'metrics/oof/{s}_{d}.csv.gz',oof_cols,rows)
    emit('metrics/per_query_oof.csv',Q_COLS,perquery)
    emit('metrics/draw_oof_summary.csv',['recipe_id','draw_id','query_count','row_count','npositive','nnegative',*BASE_METRICS,'status'],drawrows)
    emit('metrics/coverage_audit.csv',list(coverage[0]),coverage)
    recipes=[]
    for s in RECIPES:
        rr=[r for r in drawrows if r['recipe_id']==s];good=all(r['status']=='COMPLETE_DESCRIPTIVE' for r in rr) and len(rr)==3
        r={'recipe_id':s,'fixed_draw_count':3,'complete_draw_count':sum(x['status']=='COMPLETE_DESCRIPTIVE' for x in rr),'sd_ddof':0,'status':'COMPLETE_DESCRIPTIVE' if good else 'INCOMPLETE'}
        if good:r.update(describe(rr,BASE_METRICS))
        recipes.append(r)
    desc_cols=[m+'_'+v for m in BASE_METRICS for v in ('mean','sd_ddof0','min','max','range')]
    emit('metrics/recipe_summary.csv',['recipe_id','fixed_draw_count','complete_draw_count','sd_ddof',*desc_cols,'status'],recipes)
    baseline={(r['draw_id'],r['epitope'],r['hla']):r for r in perquery if r['recipe_id']=='S0'};vs=[];vsdraw=[]
    for s in RECIPES:
      for d in DRAWS:
        rr=[r for r in perquery if r['recipe_id']==s and r['draw_id']==d];local=[]
        complete=next(x for x in drawrows if x['recipe_id']==s and x['draw_id']==d)['status']=='COMPLETE_DESCRIPTIVE' and next(x for x in drawrows if x['recipe_id']=='S0' and x['draw_id']==d)['status']=='COMPLETE_DESCRIPTIVE'
        for r in rr:
            b=baseline.get((d,r['epitope'],r['hla']))
            vr={'recipe_id':s,'draw_id':d,'epitope':r['epitope'],'hla':r['hla'],'pmhc_id':r['pmhc_id'],'recipe_fold_id':r['fold_id'],'S0_fold_id':b['fold_id'] if b else '', 'npositive':r['npositive'],'nnegative':r['nnegative'],'status':'COMPLETE_DESCRIPTIVE' if complete and b else 'INCOMPLETE'}
            if complete and b:vr.update({'delta_G2_S0_'+m:float(r[m])-float(b[m]) for m in BASE_METRICS})
            local.append(vr)
        vs.extend(local);dr={'recipe_id':s,'draw_id':d,'paired_query_count':len(local),'status':'COMPLETE_DESCRIPTIVE' if complete else 'INCOMPLETE'}
        if complete:dr.update(means(local,DELTA_COLS))
        vsdraw.append(dr)
    emit('metrics/vs_s0_per_query.csv',['recipe_id','draw_id','epitope','hla','pmhc_id','recipe_fold_id','S0_fold_id','npositive','nnegative',*DELTA_COLS,'status'],vs)
    emit('metrics/vs_s0_draw_summary.csv',['recipe_id','draw_id','paired_query_count',*DELTA_COLS,'status'],vsdraw)
    vsrecipe=[]
    for s in RECIPES:
        rr=[r for r in vsdraw if r['recipe_id']==s];good=all(r['status']=='COMPLETE_DESCRIPTIVE' for r in rr)
        r={'recipe_id':s,'fixed_draw_count':3,'complete_draw_count':sum(x['status']=='COMPLETE_DESCRIPTIVE' for x in rr),'sd_ddof':0,'status':'COMPLETE_DESCRIPTIVE' if good else 'INCOMPLETE'}
        if good:r.update(describe(rr,DELTA_COLS))
        vsrecipe.append(r)
    emit('metrics/vs_s0_recipe_summary.csv',['recipe_id','fixed_draw_count','complete_draw_count','sd_ddof',*[m+'_'+v for m in DELTA_COLS for v in ('mean','sd_ddof0','min','max','range')],'status'],vsrecipe)
    for rel,obj in [('metrics/input_identity.json',{'schema_version':1,'g2_manifest_sha256':G2_SHA,'inputs':sorted(inputs.values(),key=lambda x:x['path'])}),('metrics/metric_run_summary.json',{'schema_version':1,'status':'PASS_DESCRIPTIVE_COMPLETE' if not errors and all(r['status']=='COMPLETE_DESCRIPTIVE' for r in drawrows) else 'INCOMPLETE','task_count':105,'passing_tasks':sum(r['status']=='PASS' for r in audits),'expected_queries_per_oof':len(common_q),'expected_positives_per_oof':len(common),'draw_summary_count':21,'recipe_summary_count':7,'sd_ddof':0,'independent_validation_status':'PENDING','model_prediction_roundtrip_status':'SEPARATE_PARENT_VALIDATION_REQUIRED','errors':errors})]:
        p=root/rel;json_write(p,obj);outputs.append({'role':'METRIC_IDENTITY_OR_STATUS','path':rel,'bytes':p.stat().st_size,'sha256':sha(p)})
    json_write(metrics/'output_catalog.json',{'schema_version':1,'files':outputs})
    print(json.dumps({'status':'PASS_DESCRIPTIVE_COMPLETE' if not errors and all(r['status']=='COMPLETE_DESCRIPTIVE' for r in drawrows) else 'INCOMPLETE','passing_tasks':sum(r['status']=='PASS' for r in audits),'outputs':len(outputs)},ensure_ascii=False))
    return 0 if not errors and all(r['status']=='COMPLETE_DESCRIPTIVE' for r in drawrows) else 2

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--g2-root',type=Path,required=True);a=p.parse_args()
    raise SystemExit(run(a.root.resolve(),a.g2_root.resolve()))
