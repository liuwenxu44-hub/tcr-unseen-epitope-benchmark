#!/usr/bin/env python3
"""Independent stdlib-only reconstruction of R04 G2 descriptive metrics.
Does not import the producer and does not train or score serialized models.
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, itertools, json, math, traceback
from collections import Counter,defaultdict
from pathlib import Path

S=tuple('S'+str(i) for i in range(7)); F=tuple('fold_'+str(i) for i in range(1,6));D=tuple('draw_'+str(i) for i in range(1,4))
M=['interaction_AP','interaction_AUROC','additive_AP','additive_AUROC','finite_no_ties_expected_AP','finite_no_ties_expected_AUROC','delta_Additive_AP','delta_Additive_AUROC','delta_finite_no_ties_AP','delta_finite_no_ties_AUROC']
V=['delta_G2_S0_'+m for m in M]
G2_SHA='98bfca307c68f01f6980f3f2f0d6bf01529254cd554ffff4a253d6fc54942d35'
COLS=['recipe_id','fold_id','draw_id','partition','model_name','model_id','model_sha256','input_source_file','input_source_sha256','input_source_row_index','tcr_cdr3_beta','epitope','hla','pmhc_id','label','probability','logit']

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while True:
            b=f.read(1048576)
            if not b:break
            h.update(b)
    return h.hexdigest()
def readcsv(p):
    op=gzip.open if str(p).endswith('.gz') else open
    with op(p,'rt',encoding='utf-8',newline='') as f:
        r=csv.DictReader(f);return r.fieldnames,list(r)
def near(a,b):
    try:return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b))<=1e-12
    except (ValueError,TypeError):return False
def q(r):return r['epitope'],r['hla']
def k(r):return r['tcr_cdr3_beta'],r['epitope'],r['hla'],int(r['label'])
def finite(p,n):
    h=math.fsum(1/i for i in range(1,n+1));return h/n+(p-1)*(n-h)/(n*(n-1))
def manual(y,score):
    # AP: add recall mass of each complete descending score group times precision.
    groups=defaultdict(lambda:[0,0])
    for label,value in zip(y,score):groups[value][label]+=1
    p=sum(y);n=len(y)-p;tp=0;seen=0;ap=[]
    for value in sorted(groups,reverse=True):
        neg,pos=groups[value];tp+=pos;seen+=pos+neg
        ap.append(pos/p*tp/seen)
    # AUROC: positive wins over all preceding lower negatives, ties half credit.
    lowerneg=0;win=[]
    for value in sorted(groups):
        neg,pos=groups[value];win.append(pos*(lowerneg+0.5*neg));lowerneg+=neg
    return math.fsum(ap),math.fsum(win)/(p*n)
def avg(rr,cols):return {c:math.fsum(float(r[c]) for r in rr)/len(rr) for c in cols}
def descriptive(rr,cols):
    out={}
    for c in cols:
        values=[float(r[c]) for r in rr];mu=math.fsum(values)/len(values)
        out[c+'_mean']=mu;out[c+'_sd_ddof0']=math.sqrt(math.fsum((v-mu)**2 for v in values)/len(values));out[c+'_min']=min(values);out[c+'_max']=max(values);out[c+'_range']=max(values)-min(values)
    return out

class Audit:
    def __init__(self):self.checks=[]
    def check(self,name,good,detail=''):
        self.checks.append({'check_id':name,'status':'PASS' if good else 'FAIL','detail':str(detail)})
        return bool(good)
    def require(self,name,good,detail=''):
        if not self.check(name,good,detail):raise ValueError(name+': '+str(detail))

def validate(root,g2,a):
    a.require('g2_manifest_identity',digest(g2/'g2_manifest.json')==G2_SHA)
    gm=json.loads((g2/'g2_manifest.json').read_text()); hashes={x['path']:x['sha256'] for x in gm['files']}
    frozen=json.loads((root/'metrics/code_contract_manifest.json').read_text())
    for x in frozen['files']:a.require('frozen_code:'+x['path'],digest(root/x['path'])==x['sha256'])
    contract=json.loads((root/'metrics/calculation_contract.json').read_text())
    expected_params=json.loads((root/'authority/model_contract.json').read_text())['actual_estimator_params']
    a.require('fixed_ddof_and_draw_definition',contract['descriptive_variation']['sd_ddof']==0 and contract['draws']==list(D) and contract['folds']==list(F))
    # Finite uniform ranking reference, exhaustive label positions for n<=7.
    for n in range(2,8):
      for p in range(1,n):
        vals=[]
        for positives in itertools.combinations(range(n),p):
            y=[int(i in positives) for i in range(n)]
            ap,_=manual(y,list(range(n,0,-1)));vals.append(ap)
        a.require(f'finite_uniform_rank_n{n}_p{p}',abs(math.fsum(vals)/len(vals)-finite(p,n))<=2e-15)
    a.require('one_positive_one_negative_finite_AP',finite(1,2)==0.75)
    a.require('constant_ties_metrics',manual([1,1,0,0],[.5]*4)==(.5,.5))
    a.require('mixed_ties_AP_AUC',all(near(x,y) for x,y in zip(manual([1,0,1,0],[.9,.9,.2,.1]),(7/12,5/8))))
    for x in json.loads((root/'metrics/output_catalog.json').read_text())['files']:
        a.require('output_hash:'+x['path'],(root/x['path']).stat().st_size==x['bytes'] and digest(root/x['path'])==x['sha256'])
    for x in json.loads((root/'metrics/input_identity.json').read_text())['inputs']:
        p=Path(x['path']);a.require('consumed_input_hash:'+x['path'],p.stat().st_size==x['bytes'] and digest(p)==x['sha256'])
    common=readcsv(g2/'execution/common_positives.csv')[1];common_keys=Counter(map(k,common));common_q={q(r) for r in common}
    a.require('common_positive_source_identity',digest(g2/'execution/common_positives.csv')==hashes['execution/common_positives.csv'])
    _,specrows=readcsv(g2/'execution/input_results.csv');spec={(r['recipe_id'],r['fold_id'],r['partition'],r['draw_id']):r for r in specrows}
    a.require('exact315_source_partition_keys',len(spec)==len(specrows)==315 and set(spec)=={(s,f,p,d) for s in S for f in F for p in ('train','validation','test') for d in D})
    li=json.loads((root/'execution/fit_ledger.json').read_text());li=li['tasks'] if isinstance(li,dict) else li
    ai=json.loads((root/'additive/artifact_index.json').read_text());ai=ai['artifacts'] if isinstance(ai,dict) else ai
    ledger={(r['recipe_id'],r['fold_id'],r['draw_id']):r for r in li};ad={(r['recipe_id'],r['fold_id'],r['draw_id']):r for r in ai}
    universe={(s,f,d) for s in S for f in F for d in D}
    a.require('exact105_interaction_task_index',set(ledger)==universe and len(li)==105)
    a.require('exact105_additive_artifact_index',set(ad)==universe and len(ai)==105)
    expected_q=[];expected_draw=[];sources=[]
    for s in S:
      for d in D:
        raw=[]
        for f in F:
            tid=f'{s}__{f}__{d}';l=ledger[s,f,d];add=ad[s,f,d]
            a.require('task_pass:'+tid,l['status']=='PASS' and add['status']=='PASS' and l['task_id']==tid and add['task_id']==tid)
            a.require('model_bytes:'+tid,digest(root/l['model_file'])==l['model_sha256'])
            a.require('prediction_bytes:'+tid,digest(root/l['prediction_file'])==l['prediction_sha256'])
            a.require('additive_bytes:'+tid,digest(root/add['artifact_file'])==add['artifact_sha256'])
            a.require('additive_reload_evidence:'+tid,bool(add.get('reload_checks')) and all(value is True for value in add['reload_checks'].values()))
            for part in ('train','validation','test'):
                ac=add['partition_checks'][part];sp=spec[s,f,part,d]
                a.require('additive_partition_evidence:'+tid+'/'+part,int(ac['rows'])==int(sp['positive_count'])+int(sp['negative_count']) and all(ac.get(name) is True for name in ('logit_exact_zero','probability_exact_half','feature_dimension_matches')))
            meta=json.loads((root/l['metadata_file']).read_text())
            ident={'recipe_id':s,'fold_id':f,'draw_id':d,'task_id':tid,'model_id':f'R04G2:{s}:{f}:{d}:orthogonalized_interaction','model_name':'orthogonalized_interaction','model_file':l['model_file'],'model_sha256':l['model_sha256']}
            a.require('model_metadata:'+tid,all(meta.get(name)==value for name,value in ident.items()))
            a.require('frozen_estimator_parameters:'+tid,all(meta['params'].get(name)==value and l['params'].get(name)==value for name,value in expected_params.items()))
            a.require('train_only_fit_partitions:'+tid,meta['model_fit_partitions']==['train'] and meta['feature_fit_partitions']==['train'])
            for part in ('train','validation','test'):
                sp=spec[s,f,part,d];mi=meta['source_inputs'][part]
                ok=True
                for typ in ('positive','negative'):
                    rel=sp[typ+'_file'];ok &= mi[typ+'_file']==rel and mi[typ+'_sha256']==hashes[rel] and int(mi[typ+'_rows'])==int(sp[typ+'_count'])
                ok &= str(mi['negative_derived_seed'])==str(sp['derived_seed'])
                a.require('metadata_source_inputs:'+tid+'/'+part,ok)
            cols,pred=readcsv(root/l['prediction_file']);a.require('prediction_schema:'+tid,cols==COLS)
            source=[];sp=spec[s,f,'test',d]
            for typ in ('positive','negative'):
                rel=sp[typ+'_file'];a.require('test_source_hash:'+tid+'/'+typ,digest(g2/rel)==hashes[rel])
                for index,row in enumerate(readcsv(g2/rel)[1]):source.append((rel,index,row))
            a.require('exact_source_prediction_row_count:'+tid,len(pred)==len(source))
            ok=True
            for row,(rel,index,src) in zip(pred,source):
                score=float(row['probability']);logit=float(row['logit'])
                ok &= k(row)==k(src) and row['pmhc_id']==src['pmhc_id'] and int(row['input_source_row_index'])==index and row['input_source_file']==rel and row['input_source_sha256']==hashes[rel]
                ok &= all(row[n]==str(v) for n,v in ident.items() if n in row) and row['partition']=='test'
                ok &= math.isfinite(score) and 0<=score<=1 and math.isfinite(logit)
                expected=1/(1+math.exp(-logit)) if logit>=0 else math.exp(logit)/(1+math.exp(logit))
                ok &= near(score,expected)
            a.require('original_row_identity_and_score:'+tid,ok)
            raw.extend([{**r,'additive_model_id':add['model_id'],'additive_artifact_file':add['artifact_file'],'additive_artifact_sha256':add['artifact_sha256'],'analytic_additive_probability':'0.5','analytic_additive_logit':'0.0'} for r in pred])
        _,saved=readcsv(root/f'metrics/oof/{s}_{d}.csv.gz')
        a.require('OOF_exact_original_rows:'+s+'/'+d,len(saved)==len(raw) and all(all(r[c]==e[c] for c in COLS+['additive_model_id','additive_artifact_file','additive_artifact_sha256']) and float(r['analytic_additive_probability'])==.5 and float(r['analytic_additive_logit'])==0 for r,e in zip(saved,raw)))
        actual=Counter(k(r) for r in raw if int(r['label'])==1);groups=defaultdict(list)
        for r in raw:groups[q(r)].append(r)
        a.require('OOF_common_universe:'+s+'/'+d,actual==common_keys and set(groups)==common_q)
        a.require('OOF_query_one_test_fold:'+s+'/'+d,all(len({r['fold_id'] for r in rr})==1 for rr in groups.values()))
        local=[]
        for query,rr in sorted(groups.items()):
            y=[int(r['label']) for r in rr];sc=[float(r['probability']) for r in rr];p=sum(y);n=len(y)-p
            a.require('query_evaluable:'+s+'/'+d+'/'+str(query),p>0 and n>0)
            ap,auc=manual(y,sc);const=p/len(y);rand=finite(p,len(y));counts=Counter(sc);ties=[x for x in counts.values() if x>=2]
            r={'recipe_id':s,'draw_id':d,'fold_id':rr[0]['fold_id'],'epitope':query[0],'hla':query[1],'pmhc_id':'|'.join(query),'npositive':p,'nnegative':n,'sample_count':len(y),'unique_score_count':len(counts),'tied_group_count':len(ties),'records_in_ties':sum(ties),'max_tie_size':max(counts.values()),'interaction_AP':ap,'interaction_AUROC':auc,'additive_AP':const,'additive_AUROC':.5,'finite_no_ties_expected_AP':rand,'finite_no_ties_expected_AUROC':.5,'delta_Additive_AP':ap-const,'delta_Additive_AUROC':auc-.5,'delta_finite_no_ties_AP':ap-rand,'delta_finite_no_ties_AUROC':auc-.5,'status':'COMPLETE_DESCRIPTIVE'}
            local.append(r)
        expected_q.extend(local);expected_draw.append({'recipe_id':s,'draw_id':d,'query_count':len(groups),'row_count':len(raw),'npositive':sum(int(r['label']) for r in raw),'nnegative':sum(1-int(r['label']) for r in raw),'status':'COMPLETE_DESCRIPTIVE',**avg(local,M)})
    def table(rel,expected,keycols,numcols):
        _,saved=readcsv(root/rel);actual={tuple(r[c] for c in keycols):r for r in saved};want={tuple(str(r[c]) for c in keycols):r for r in expected}
        a.require('table_keys:'+rel,len(saved)==len(actual)==len(expected) and set(actual)==set(want))
        for key,e in want.items():
            r=actual[key];ok=True
            for col,val in e.items():
                ok &= near(r.get(col),val) if col in numcols else r.get(col)==str(val)
            a.check('table_values:'+rel+'/'+','.join(key),ok)
    counts=['npositive','nnegative','sample_count','unique_score_count','tied_group_count','records_in_ties','max_tie_size']
    table('metrics/per_query_oof.csv',expected_q,['recipe_id','draw_id','epitope','hla'],M+counts)
    table('metrics/draw_oof_summary.csv',expected_draw,['recipe_id','draw_id'],M+['query_count','row_count','npositive','nnegative'])
    recipes=[]
    for s in S:recipes.append({'recipe_id':s,'fixed_draw_count':3,'complete_draw_count':3,'sd_ddof':0,'status':'COMPLETE_DESCRIPTIVE',**descriptive([r for r in expected_draw if r['recipe_id']==s],M)})
    table('metrics/recipe_summary.csv',recipes,['recipe_id'],[m+'_'+z for m in M for z in ('mean','sd_ddof0','min','max','range')]+['fixed_draw_count','complete_draw_count','sd_ddof'])
    base={(r['draw_id'],r['epitope'],r['hla']):r for r in expected_q if r['recipe_id']=='S0'};vs=[];vd=[]
    for s in S:
      for d in D:
        rr=[]
        for r in [r for r in expected_q if r['recipe_id']==s and r['draw_id']==d]:
            b=base[d,r['epitope'],r['hla']];v={'recipe_id':s,'draw_id':d,'epitope':r['epitope'],'hla':r['hla'],'pmhc_id':r['pmhc_id'],'recipe_fold_id':r['fold_id'],'S0_fold_id':b['fold_id'],'npositive':r['npositive'],'nnegative':r['nnegative'],'status':'COMPLETE_DESCRIPTIVE',**{'delta_G2_S0_'+m:r[m]-b[m] for m in M}}
            rr.append(v)
        vs.extend(rr);vd.append({'recipe_id':s,'draw_id':d,'paired_query_count':len(rr),'status':'COMPLETE_DESCRIPTIVE',**avg(rr,V)})
    table('metrics/vs_s0_per_query.csv',vs,['recipe_id','draw_id','epitope','hla'],V+['npositive','nnegative'])
    table('metrics/vs_s0_draw_summary.csv',vd,['recipe_id','draw_id'],V+['paired_query_count'])
    vr=[{'recipe_id':s,'fixed_draw_count':3,'complete_draw_count':3,'sd_ddof':0,'status':'COMPLETE_DESCRIPTIVE',**descriptive([r for r in vd if r['recipe_id']==s],V)} for s in S]
    table('metrics/vs_s0_recipe_summary.csv',vr,['recipe_id'],[m+'_'+z for m in V for z in ('mean','sd_ddof0','min','max','range')]+['fixed_draw_count','complete_draw_count','sd_ddof'])
    _,coverage=readcsv(root/'metrics/coverage_audit.csv')
    a.require('coverage_exact21',len(coverage)==21 and {(r['recipe_id'],r['draw_id']) for r in coverage}=={(s,d) for s in S for d in D})
    for r in coverage:
        a.check('coverage_complete:'+r['recipe_id']+'/'+r['draw_id'],r['status']=='COMPLETE_DESCRIPTIVE' and int(r['complete_test_folds'])==5 and r['common_positive_coverage_exact']=='True' and r['common_query_coverage_exact']=='True' and int(r['queries_in_multiple_test_folds'])==0 and int(r['balanced_query_count'])==len(common_q) and int(r['observed_positive_count'])==len(common) and int(r['observed_query_count'])==len(common_q))
    run=json.loads((root/'metrics/metric_run_summary.json').read_text());a.check('producer_complete_status',run['status']=='PASS_DESCRIPTIVE_COMPLETE' and run['passing_tasks']==105 and run['sd_ddof']==0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--g2-root',type=Path,required=True);p.add_argument('--attempt',default='1');args=p.parse_args();a=Audit()
    out=args.root/'validation';out.mkdir(exist_ok=True)
    cp=out/f'metric_checks_attempt{args.attempt}.csv';jp=out/f'metric_validation_attempt{args.attempt}.json'
    if cp.exists() or jp.exists():raise FileExistsError('Refuse overwrite validation attempt')
    try:validate(args.root.resolve(),args.g2_root.resolve(),a)
    except Exception as exc:a.check('fatal_validation_error',False,type(exc).__name__+': '+str(exc))
    with cp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['check_id','status','detail'],lineterminator='\n');w.writeheader();w.writerows(a.checks)
    fail=[r for r in a.checks if r['status']=='FAIL'];obj={'schema_version':1,'status':'PASS' if not fail else 'FAIL','check_count':len(a.checks),'passing_checks':len(a.checks)-len(fail),'failure_count':len(fail),'failures':fail,'checks_file':cp.name,'checks_sha256':digest(cp),'g2_manifest_sha256':G2_SHA,'independent_algorithm':'stdlib manual complete-score-group AP and Mann-Whitney AUROC, rebuilt from original 105 test prediction files; no producer imports','numeric_abs_tolerance':1e-12,'numeric_rel_tolerance':0,'model_roundtrip':'NOT_CLAIMED_SEPARATE_PARENT_VALIDATION_REQUIRED','descriptive_only':True}
    jp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:obj[k] for k in ('status','check_count','failure_count')}));raise SystemExit(0 if not fail else 2)
