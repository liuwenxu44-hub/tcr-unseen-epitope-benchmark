"""Execute only original H1 capacity, rematch and final structural audits.

All scientific parents are mounted read-only. No model or score code is loaded.
The legacy sampler solver, original seed derivation, row tagging and CSV encoder
are imported without edits. Forbidden membership is the sole changed rule.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,io,json,platform,sys,traceback
from pathlib import Path
from collections import Counter,defaultdict
from itertools import combinations
from functools import lru_cache
from datetime import datetime,timezone
import numpy,scipy
from tcrue.degree_matched_negatives import solve_degree_matched_partition,audit_degree_matched_partition,ExactDegreeMatchError,_csv_text
from tcrue.degree_matched_resamples import derive_partition_seed,_tag_resample_rows
from tcrue.negative_sampling import load_positive_registry

ROLES=('train','validation','test')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def obj(p):return json.loads(p.read_text())
def read(p):
    with (gzip.open(p,'rt',newline='') if p.suffix=='.gz' else p.open(newline='')) as f:
        r=csv.DictReader(f);return list(r.fieldnames),list(r)
def key(r):return tuple(str(r[k]).strip().upper() for k in ('tcr_cdr3_beta','epitope','hla'))
def canonical_sha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def edge_sha(rows):
    ks=[key(r) for r in rows];assert len(ks)==len(set(ks));return canonical_sha(sorted(ks))
def sanitized(v):
    if isinstance(v,dict):return {k.replace('known_positive_registry','forbidden_registry'):sanitized(x) for k,x in v.items()}
    if isinstance(v,list):return [sanitized(x) for x in v]
    return v
@lru_cache(None)
def similarity(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return 1.0-prev[-1]/max(len(a),len(b))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--r05',type=Path,required=True);ap.add_argument('--g2',type=Path,required=True);ap.add_argument('--generation',type=Path,required=True);a=ap.parse_args()
    w=a.work;roots={'source':a.source,'r05':a.r05,'g2':a.g2,'generation':a.generation};files=[]
    def track(p,role):files.append(dict(path=p.relative_to(w).as_posix(),role=role,sha256=sha(p),bytes=p.stat().st_size))
    def writej(name,value,role='H1_STRUCTURAL_AUDIT'):
        p=w/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
        track(p,role);return name
    def writecsv(name,rows,fields=None,role='H1_STRUCTURAL_TABLE'):
        p=w/name;p.parent.mkdir(parents=True,exist_ok=True);fields=list(rows[0]) if fields is None else fields
        if p.suffix=='.gz':
            with p.open('xb') as raw:
                with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as z:
                    with io.TextIOWrapper(z,encoding='utf-8',newline='') as f:
                        c=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');c.writeheader();c.writerows(rows)
        else:
            with p.open('x',newline='') as f:
                c=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');c.writeheader();c.writerows(rows)
        track(p,role);return name
    def bytecopy(src,rel,expected,role):
        assert sha(src)==expected,str(src);p=w/rel;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(src.read_bytes())
        assert sha(p)==expected;track(p,role)
    def stop(reason,details):
        writej('structural_gate.json',dict(status='STOP_H1_STRUCTURE_NO_PARTIAL_RELEASE',reason=reason,details=details,future_model_execution_eligible=False,training_fits=0,models_loaded=0,scores_read=0,performance_computed=False,finished_utc=datetime.now(timezone.utc).isoformat()))
        writej('owned_catalog.json',dict(status='STOP',files=files.copy()))
        print(json.dumps(dict(status='STOP',reason=reason,details=details)),flush=True);raise SystemExit(2)
    try:
        freeze=obj(w/'code_rules_input_freeze.json');assert freeze['status']=='FROZEN_BEFORE_ANY_H1_REMATCH'
        for r in freeze['files']:
            p=w/r['path'];assert sha(p)==r['sha256'] and p.stat().st_size==r['bytes'],r['path']
            track(p,r['role'])
        track(w/'code_rules_input_freeze.json','PREMATCH_CODE_RULES_INPUT_FREEZE')
        allowed={r['path'] for r in freeze['files']}|{'code_rules_input_freeze.json','execution_stdout.log','scratch'}
        assert {p.name for p in w.iterdir()}<=allowed,'Unexpected structure artifacts before execution'
        sourcechecks=[]
        for r in freeze['selected_input_files']:
            p=roots[r['root_id']]/r['path'];observed=sha(p);passed=observed==r['sha256'] and p.stat().st_size==r['bytes']
            sourcechecks.append(dict(**r,observed_sha256=observed,status='PASS' if passed else 'FAIL'));assert passed,str(p)
        writej('input_identity_before.json',dict(status='PASS_ALL_EXPLICIT_ORIGINAL_AND_AUTHORITY_INPUTS',files=sourcechecks))
        contract=obj(w/'contract.json');_,tasks=read(w/'task75.csv');_,capacities=read(w/'capacity15.csv')
        for r in tasks:
            for k in ['base_seed','derived_seed','positive_rows','old_negative_rows','old_forbidden135555_keys','old_unknown_keys']:r[k]=int(r[k])
        assert len(tasks)==75 and len(capacities)==15 and sum(t['action']=='REMATCH_ORIGINAL_SAMPLER' for t in tasks)==42
        _,registryrows=read(a.g2/contract['forbidden_registry_path']);registry={key(r) for r in registryrows}
        assert len(registryrows)==len(registry)==135555
        reg_lf_sha=hashlib.sha256(('\n'.join(sorted('|'.join(k) for k in registry))+'\n').encode()).hexdigest()
        assert reg_lf_sha==obj(a.g2/'registry/registry_key_identity.json')['key_sha256']
        original=obj(a.source/contract['original_sampler_manifest_path']);oldregistry={tuple(str(v).strip().upper() for v in k) for k in load_positive_registry(a.source/contract['original_registry_path'])};assert len(oldregistry)==117615 and oldregistry<=registry
        unknown={key(r) for r in registryrows if r.get('evidence_class')=='CONSERVATIVE_RISK_EXCLUSION_UNKNOWN'}
        del registryrows
        writej('runtime_identity.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__,science_image=contract['science_image'],code_rules_input_freeze_sha256=sha(w/'code_rules_input_freeze.json'),contract_sha256=sha(w/'contract.json'),forbidden_registry_sha256=sha(a.g2/contract['forbidden_registry_path']),forbidden_registry_exact_keys=135555,forbidden_registry_key_lf_sha256=reg_lf_sha,original_registry_keys=len(oldregistry),training_fits=0,models_loaded=0,scores_read=0,performance_computed=False))
        positives={};graphs={};capacityaudit=[]
        # All15 whole-graph capacity checks precede original sanity or any rematch.
        for i,c in enumerate(capacities,1):
            fold,role=c['fold_id'],c['role'];_,ps=read(a.source/c['positive_source_path']);positives[(fold,role)]=ps
            assert sha(a.source/c['positive_source_path'])==c['positive_sha256'] and len(ps)==int(c['positive_rows']) and all(str(r['label'])=='1' for r in ps)
            pk={key(r) for r in ps};assert len(pk)==len(ps) and pk<=registry
            ts=Counter(k[0] for k in pk);qs=Counter(k[1:] for k in pk);h=hashlib.sha256();candidate=0
            for t in sorted(ts):
                for e,l in sorted(qs):
                    if (t,e,l) not in registry:h.update((t+'\t'+e+'\t'+l+'\n').encode('ascii'));candidate+=1
            graph=dict(allowed_edge_set_sha256=h.hexdigest(),candidate_edge_count=candidate,registry_rejected_candidate_count=len(ts)*len(qs)-candidate,positive_tcr_degree_sha256=canonical_sha(sorted(ts.items())),positive_full_query_degree_sha256=canonical_sha(sorted(qs.items())),positive_edge_set_sha256=edge_sha(ps),positive_file_sha256=c['positive_sha256'])
            graph['capacitated_graph_sha256']=canonical_sha(graph);graphs[(fold,role)]=graph
            try:
                witness,audit=solve_degree_matched_partition(ps,registry,fold_id=fold,partition=role,seed=None);del witness
            except ExactDegreeMatchError as e:
                capacityaudit.append(dict(capacity_problem_id=c['capacity_problem_id'],fold_id=fold,role=role,status='MATCHING_INFEASIBLE',failure=e.report,**graph));writej('capacity15.json',capacityaudit);stop('FULL_ORIGINAL_CAPACITY_PROBLEM_INFEASIBLE',capacityaudit[-1])
            assert audit['maximum_flow_value']==len(ps) and audit['candidate_edge_count']==candidate
            capacityaudit.append(dict(capacity_problem_id=c['capacity_problem_id'],fold_id=fold,role=role,status='PASS',maximum_flow_value=audit['maximum_flow_value'],required_edges=len(ps),positive_tcr_count=len(ts),positive_full_query_count=len(qs),original_quotas_preserved=True,full_complete_graph_maxflow=True,solver_audit=sanitized(audit),**graph))
            print(f'Original full capacity PASS {i}/15 {fold} {role}: {len(ps)} edges',flush=True)
        writej('capacity15.json',capacityaudit);writecsv('capacity15_summary.csv',[{k:v for k,v in x.items() if k!='solver_audit'} for x in capacityaudit]);writej('capacity_gate.json',dict(status='PASS_ALL15_ORIGINAL_COMPLETE_GRAPH_CAPACITIES',passed=15,requested=15,actual_full_graph_maxflow_calls=15,rematch_started=False))
        print('PASS_ALL15_ORIGINAL_COMPLETE_GRAPH_CAPACITIES',flush=True)
        # A bounded three-role original-data reproduction establishes unchanged
        # legacy sampler+encoder behavior without re-auditing historical outputs.
        sanity=[]
        for tid in contract['sanity_reproduction_tasks']:
            t=next(r for r in tasks if r['task_id']==tid);fold,role,draw=t['fold_id'],t['role'],t['draw_id'];derived=derive_partition_seed(t['base_seed'],fold,role);assert derived==t['derived_seed']
            rows,old_audit=solve_degree_matched_partition(positives[(fold,role)],oldregistry,fold_id=fold,partition=role,seed=derived)
            rows=_tag_resample_rows(rows,draw);raw=_csv_text(rows).encode();observed=hashlib.sha256(raw).hexdigest()
            result=dict(task_id=tid,base_seed=t['base_seed'],derived_seed=derived,expected_original_sha256=t['old_negative_sha256'],reproduced_sha256=observed,status='PASS' if observed==t['old_negative_sha256'] else 'FAIL')
            sanity.append(result)
            if result['status']!='PASS':writej('original_sampler_sanity3.json',sanity);stop('ORIGINAL_SAMPLER_REPRODUCTION_FAILED',result)
        writej('original_sampler_sanity3.json',sanity)
        # Freeze/copy the original15 positives verbatim only after the full gate.
        for c in capacities:bytecopy(a.source/c['positive_source_path'],c['positive_output_path'],c['positive_sha256'],'ORIGINAL_POSITIVE_UNCHANGED_BYTES')
        actions=[];rematchcalls=0;reusecount=0
        for i,t in enumerate(tasks,1):
            fold,role,draw=t['fold_id'],t['role'],t['draw_id'];ps=positives[(fold,role)];_,oldneg=read(a.source/t['negative_source_path']);oldks={key(r) for r in oldneg};oldbad=oldks&registry
            assert len(oldks)==len(oldneg)==t['old_negative_rows']==len(ps) and len(oldbad)==t['old_forbidden135555_keys']
            assert derive_partition_seed(t['base_seed'],fold,role)==t['derived_seed']==original['folds'][fold]['partitions'][role]['resamples'][draw]['derived_seed']
            if bool(oldbad)!=(t['action']=='REMATCH_ORIGINAL_SAMPLER'):stop('FROZEN42_33_ACTION_BUDGET_CHANGED',t)
            if t['action']=='REMATCH_ORIGINAL_SAMPLER':
                try:rows,audit=solve_degree_matched_partition(ps,registry,fold_id=fold,partition=role,seed=t['derived_seed'])
                except ExactDegreeMatchError as e:stop('SEEDED_ORIGINAL_REMATCH_INFEASIBLE',dict(task_id=t['task_id'],report=e.report))
                rematchcalls+=1;rows=_tag_resample_rows(rows,draw);p=w/t['negative_output_path'];p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('xb') as f:f.write(_csv_text(rows).encode())
                track(p,'REPAIRED_ORIGINAL_NEGATIVE_GRAPH');action='REMATCH'
            else:
                bytecopy(a.source/t['negative_source_path'],t['negative_output_path'],t['old_negative_sha256'],'CONDITIONALLY_REUSED_CLEAN_ORIGINAL_GRAPH_BYTES');rows=oldneg;reusecount+=1;action='VERIFIED_BYTE_REUSE'
            audit=sanitized(audit_degree_matched_partition(ps,rows,registry));newks={key(r) for r in rows}
            assert len(newks)==len(rows)==len(ps) and not(newks&registry) and all(str(r['label'])=='0' for r in rows)
            diff=[dict(change='REMOVED',tcr_cdr3_beta=k[0],epitope=k[1],hla=k[2],was_conservative_forbidden=k in registry) for k in sorted(oldks-newks)]+[dict(change='ADDED',tcr_cdr3_beta=k[0],epitope=k[1],hla=k[2],was_conservative_forbidden=k in registry) for k in sorted(newks-oldks)]
            diffpath=writecsv('edge_diffs/'+t['task_id']+'.csv.gz',diff,['change','tcr_cdr3_beta','epitope','hla','was_conservative_forbidden'])
            result=dict(graph_id=t['task_id'],resample_id=draw,fold_id=fold,partition=role,role=role,action=action,base_seed=t['base_seed'],derived_seed=t['derived_seed'],positive_source_path=t['positive_source_path'],positive_sha256=t['positive_sha256'],positive_rows=len(ps),negative_source_path=t['negative_source_path'],old_negative_sha256=t['old_negative_sha256'],negative_output_path=t['negative_output_path'],new_negative_sha256=sha(w/t['negative_output_path']),old_edge_set_sha256=edge_sha(oldneg),new_edge_set_sha256=edge_sha(rows),old_forbidden_keys=len(oldbad),old_unknown_keys=t['old_unknown_keys'],removed_old_forbidden_keys=len(oldbad-newks),retained_old_forbidden_keys=len(oldbad&newks),removed_edges=len(oldks-newks),added_edges=len(newks-oldks),retained_edges=len(oldks&newks),byte_identical_to_original=sha(w/t['negative_output_path'])==t['old_negative_sha256'],edge_diff_path=diffpath,immediate_degree_forbid_duplicate_audit=audit)
            if action=='VERIFIED_BYTE_REUSE':assert result['byte_identical_to_original'] and result['removed_edges']==result['added_edges']==0
            else:assert not result['byte_identical_to_original'] and result['retained_old_forbidden_keys']==0
            actions.append(result)
            if i%15==0:print(f'Original generation {i}/75: rematched {rematchcalls}/42, byte reuse {reusecount}/33',flush=True)
        assert rematchcalls==42 and reusecount==33
        assert Counter(x['role'] for x in actions if x['action']=='REMATCH')==contract['rematch_by_role']
        writej('original_new_graph_diff75.json',actions);writecsv('original_new_graph_diff75.csv',[{k:v for k,v in x.items() if k!='immediate_degree_forbid_duplicate_audit'} for x in actions])
        # Every75 final input gets a new actual full graph flow plus independent
        # exact degrees, forbidden membership, duplicates and locality checks.
        final=[];finalsets={};index=[];finalcalls=0
        serverw='/data/lwx/proj/TCR_unseen_epitope/runs/revision_20261008_v1_1/r05_original_repair_h1_20261009/structure'
        for i,t in enumerate(tasks,1):
            fold,role,draw=t['fold_id'],t['role'],t['draw_id'];ps=positives[(fold,role)];_,ns=read(w/t['negative_output_path']);ks={key(r) for r in ns}
            witness,flow=solve_degree_matched_partition(ps,registry,fold_id=fold,partition=role,seed=t['derived_seed']);finalcalls+=1
            independent=sanitized(audit_degree_matched_partition(ps,ns,registry));assert flow['maximum_flow_value']==len(ps)
            if t['action']=='REMATCH_ORIGINAL_SAMPLER':assert hashlib.sha256(_csv_text(_tag_resample_rows(witness,draw)).encode()).hexdigest()==sha(w/t['negative_output_path'])
            del witness
            assert len(ks)==len(ns)==len(ps) and not ks&registry and all(str(r['label'])=='0' for r in ns)
            assert all(r['fold_id']==fold and r['partition']==role and r['pmhc_id']=='|'.join(key(r)[1:]) for r in ns)
            tp=Counter(key(r)[0] for r in ps);tn=Counter(k[0] for k in ks);qp=Counter(key(r)[1:] for r in ps);qn=Counter(k[1:] for k in ks)
            degrees=[dict(entity_type='TCR_CDR3beta',entity_id=k,positive_degree=tp[k],negative_degree=tn[k],difference=tn[k]-tp[k],exact=tp[k]==tn[k]) for k in sorted(set(tp)|set(tn))]+[dict(entity_type='FULL_QUERY',entity_id='|'.join(k),positive_degree=qp[k],negative_degree=qn[k],difference=qn[k]-qp[k],exact=qp[k]==qn[k]) for k in sorted(set(qp)|set(qn))]
            assert all(x['exact'] for x in degrees)
            dpath=writecsv('degree_audits/'+t['task_id']+'.csv.gz',degrees)
            item=dict(graph_id=t['task_id'],resample_id=draw,fold_id=fold,partition=role,status='PASS',full_complete_graph_maxflow=True,maximum_flow_value=flow['maximum_flow_value'],required_edges=len(ps),base_seed=t['base_seed'],derived_seed=t['derived_seed'],negative_sha256=sha(w/t['negative_output_path']),negative_edge_set_sha256=edge_sha(ns),negative_tcr_degree_sha256=canonical_sha(sorted(tn.items())),negative_full_query_degree_sha256=canonical_sha(sorted(qn.items())),degree_audit_path=dpath,independent_audit=independent,flow_audit=sanitized(flow),**graphs[(fold,role)])
            final.append(item);finalsets[(draw,fold,role)]=ks
            action='REMATCH' if t['action']=='REMATCH_ORIGINAL_SAMPLER' else 'VERIFIED_BYTE_REUSE'
            index.append(dict(graph_id=t['task_id'],resample_id=draw,fold_id=fold,partition=role,role=role,positive=dict(path=serverw+'/'+t['positive_output_path'],relative_path=t['positive_output_path'],sha256=t['positive_sha256'],rows=len(ps)),negative=dict(path=serverw+'/'+t['negative_output_path'],relative_path=t['negative_output_path'],sha256=sha(w/t['negative_output_path']),rows=len(ns)),action=action,base_seed=t['base_seed'],derived_seed=t['derived_seed'],positive_path=serverw+'/'+t['positive_output_path'],positive_sha256=t['positive_sha256'],negative_path=serverw+'/'+t['negative_output_path'],negative_sha256=sha(w/t['negative_output_path']),graph_action=action))
            if i%15==0:print(f'Final complete-graph/degree/forbid/locality/duplicate audits PASS {i}/75',flush=True)
        writej('final_graph_audits75.json',final);writecsv('final_graph_audits75_summary.csv',[{k:v for k,v in x.items() if k not in ['independent_audit','flow_audit']} for x in final])
        # Recheck original peptide component and role boundaries; TCR overlap is
        # allowed by the frozen task and is recorded, never mislabeled leakage.
        split=obj(a.source/contract['original_split_manifest_path']);threshold=float(split['similarity']['threshold']);assert threshold==0.5
        leakage=[]
        for fold in ['fold_'+str(i) for i in range(1,6)]:
            psets={r:{key(x) for x in positives[(fold,r)]} for r in ROLES};peps={r:{k[1] for k in psets[r]} for r in ROLES};queries={r:{k[1:] for k in psets[r]} for r in ROLES};tcrs={r:{k[0] for k in psets[r]} for r in ROLES}
            allpeps=sorted(set.union(*peps.values()));parent={p:p for p in allpeps}
            def find(p):
                while parent[p]!=p:parent[p]=parent[parent[p]];p=parent[p]
                return p
            for l,r in combinations(allpeps,2):
                if similarity(l,r)>threshold:parent[find(l)]=find(r)
            components=defaultdict(set)
            for p in allpeps:components[find(p)].add(p)
            splitcomponents=[sorted(c) for c in components.values() if sum(bool(c&peps[r]) for r in ROLES)!=1]
            assert not splitcomponents
            for draw in ['resample_'+str(i) for i in range(1,6)]:
                nsets={r:finalsets[(draw,fold,r)] for r in ROLES};combined={r:psets[r]|nsets[r] for r in ROLES}
                overlaps={'__'.join([l,r]):len(combined[l]&combined[r]) for l,r in combinations(ROLES,2)};peptide_overlap={'__'.join([l,r]):len(peps[l]&peps[r]) for l,r in combinations(ROLES,2)};query_overlap={'__'.join([l,r]):len(queries[l]&queries[r]) for l,r in combinations(ROLES,2)}
                cross={'__'.join([l,r]):max(similarity(x,y) for x in peps[l] for y in peps[r]) for l,r in combinations(ROLES,2)}
                assert not any(overlaps.values()) and not any(peptide_overlap.values()) and not any(query_overlap.values()) and all(v<=threshold for v in cross.values())
                leakage.append(dict(resample_id=draw,fold_id=fold,status='PASS',combined_interaction_role_overlap=overlaps,peptide_role_overlap=peptide_overlap,full_query_role_overlap=query_overlap,component_count=len(components),split_component_count=0,cross_role_max_similarity=cross,frozen_threshold=threshold,positive_bytes_unchanged=True,negative_tcr_query_locality=True,TCR_overlap_allowed_original_task={'__'.join([l,r]):len(tcrs[l]&tcrs[r]) for l,r in combinations(ROLES,2)},TCR_disjointness_claimed=False))
        writej('leakage25.json',leakage)
        duplicates=[]
        for c in capacities:
            fold,role=c['fold_id'],c['role'];groups=defaultdict(list);pairs=[]
            for draw in ['resample_'+str(i) for i in range(1,6)]:groups[canonical_sha(sorted(finalsets[(draw,fold,role)]))].append(draw)
            for l,r in combinations(['resample_'+str(i) for i in range(1,6)],2):
                L,R=finalsets[(l,fold,role)],finalsets[(r,fold,role)];pairs.append(dict(left=l,right=r,status='REPEATED_DRAW_RETAINED' if L==R else 'DISTINCT_DRAW',intersection_edges=len(L&R),union_edges=len(L|R),jaccard=len(L&R)/len(L|R)))
            duplicates.append(dict(fold_id=fold,role=role,requested_draws=5,retained_draws=5,unique_edge_sets=len(groups),repeated_groups=[x for x in groups.values() if len(x)>1],seed_replacement_performed=False,statistical_independence_or_uniform_sampling_claimed=False,pairwise=pairs))
        writej('draw_repeat_audit15.json',duplicates);writecsv('draw_pairwise150.csv',[dict(fold_id=x['fold_id'],role=x['role'],**p) for x in duplicates for p in x['pairwise']])
        after=[]
        for r in freeze['selected_input_files']:
            p=roots[r['root_id']]/r['path'];observed=sha(p);assert observed==r['sha256'] and p.stat().st_size==r['bytes'],str(p);after.append(dict(**r,observed_sha256=observed,status='PASS'))
        writej('input_identity_after.json',dict(status='PASS_ALL_EXPLICIT_PARENT_AND_ORIGINAL_INPUTS_UNCHANGED',files=after))
        writej('input_index75.json',index,'H1_TRAINING_SCORING_HANDOFF75')
        summary=dict(status='PASS_H1_ORIGINAL_STRUCTURE_ALL75',capacity_problems=15,passing_capacity_problems=15,capacity_full_graph_maxflow_calls=15,original_sampler_sanity_reproduction_calls=3,original_sampler_sanity_reproductions_passed=3,rematch_graphs=42,rematch_by_role=dict(Counter(x['role'] for x in actions if x['action']=='REMATCH')),byte_reused_graphs=33,final_graphs=75,final_graph_full_maxflow_calls=finalcalls,passing_final_graphs=75,passing_leakage_fold_draws=25,forbidden_registry_keys=135555,forbidden_intersections_final=0,exact_tcr_degree_graphs=75,exact_full_query_degree_graphs=75,old_forbidden_occurrences_removed=sum(x['old_forbidden_keys'] for x in actions),repeated_draw_fold_roles=sum(bool(x['repeated_groups']) for x in duplicates),positive_file_changes=0,split_component_changes=0,seed_changes=0,sampler_algorithm_changes=0,original_inputs_changed=0,actual_fit_budget_changed=False,training_fits=0,models_loaded=0,scores_read=0,performance_computed=False,future_model_execution_eligible=True,input_index75_sha256=sha(w/'input_index75.json'),pre_match_freeze_sha256=sha(w/'code_rules_input_freeze.json'),finished_utc=datetime.now(timezone.utc).isoformat())
        writej('structural_gate.json',summary);writej('owned_catalog.json',dict(status='PASS_H1_ORIGINAL_STRUCTURE_ALL75',files=files.copy()))
        print(json.dumps(summary),flush=True)
    except SystemExit:raise
    except Exception as e:
        if not(w/'structural_gate.json').exists():stop('H1_EXECUTION_OR_IDENTITY_ERROR',dict(exception_type=type(e).__name__,exception=str(e),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
