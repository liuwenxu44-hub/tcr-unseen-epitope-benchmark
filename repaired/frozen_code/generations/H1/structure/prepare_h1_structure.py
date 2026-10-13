"""Create the pre-matching H1 rules/input/code freeze; never fit or score."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
from datetime import datetime, timezone

W = Path(__file__).resolve().parent
P = W.parents[3]
V = P / 'results/revision_20261008_v1_1'
R05 = V / 'r05_decision_20261009'
G2 = V / 'r04_structure_g2_20261009'
R05_HASH = 'e263c734ab47973068fd6de013840601d9d32126bd82661fdc1055f00664f561'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def obj(p): return json.loads(p.read_text())
def read(p): return list(csv.DictReader(p.open(newline='')))
def writej(p, data):
    with p.open('x') as f: json.dump(data, f, ensure_ascii=False, indent=2); f.write('\n')
def writecsv(p, rows):
    with p.open('x', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def identity(root_id, root, rel, role, expected=None):
    p=root/rel; observed=sha(p)
    if expected is not None: assert observed==expected, str(p)
    return dict(root_id=root_id,path=rel,role=role,bytes=p.stat().st_size,sha256=observed)

def main():
    assert {p.name for p in W.iterdir()} <= {'prepare_h1_structure.py','run_h1_structure.py','launch_h1_structure.sh'}, 'Unexpected output before pre-match freeze'
    assert sha(R05/'r05_manifest.json')==R05_HASH
    r05=obj(R05/'r05_manifest.json'); assert r05['freeze_status']=='frozen'
    bind={r['path']:r for r in r05['files']}
    selected=['internal/future_minimal_repair_plan.json','internal/original15_capacity_problem_decisions.csv','internal/original_and_canonical_graph_decisions.csv','authority/parent_r04_structure_g2_20261009_manifest.json']
    sources=[identity('r05',R05,'r05_manifest.json','R05_DECISION_AUTHORITY',R05_HASH),identity('generation',W.parent,'scope_amendment.json','USER_AUTHORIZED_H1_SCOPE_AMENDMENT','3e0fc4bea2773f7b898ba80c77820d38d86a1cc81156b365688b0778106431be')]
    for rel in selected: sources.append(identity('r05',R05,rel,'R05_SELECTED_AUTHORITY',bind[rel]['sha256']))
    parent=obj(R05/selected[-1]); assert parent['freeze_status']=='frozen'
    parentbind={r['path']:r for r in parent['files']}
    assert parent['project_root_relative_to_manifest']=='.'
    for rel in ['registry/registry_manifest.json','registry/registry_policy.json','registry/forbidden_registry.csv.gz','registry/registry_key_identity.json']:
        sources.append(identity('g2',G2,rel,'IMMUTABLE_CONSERVATIVE_FORBIDDEN_REGISTRY',parentbind[rel]['sha256']))
    registrymanifest=obj(G2/'registry/registry_manifest.json'); registrybind={r['path']:r for r in registrymanifest['files']}
    assert registrymanifest['freeze_status']=='frozen'
    for rel in ['registry/registry_policy.json','registry/forbidden_registry.csv.gz','registry/registry_key_identity.json']:
        assert sha(G2/rel)==registrybind[rel]['sha256']
    plan=obj(R05/selected[0]); assert plan['original_negative_graphs_to_rematch']==42 and plan['original_negative_graphs_conditionally_reusable']==33
    assert plan['rematch_by_role']=={'train':18,'validation':11,'test':13}
    capacities=read(R05/selected[1]); graphs=[r for r in read(R05/selected[2]) if r['scope']=='internal_original_resamples']
    assert len(capacities)==15 and len(graphs)==75
    manifestrel='data/negatives/distance_aware_5fold_rebalanced_query_balanced_degree_matched_cap100_resamples/degree_matched_resamples_manifest.json'
    original=obj(P/manifestrel)
    samplerconfig=original['config'];sources.append(identity('source',P,samplerconfig['path'],'ORIGINAL_SAMPLER_CONFIGURATION',samplerconfig['sha256']))
    sources.append(identity('source',P,manifestrel,'ORIGINAL_RESAMPLE_MANIFEST'))
    for rel in ['src/tcrue/degree_matched_negatives.py','src/tcrue/degree_matched_resamples.py','src/tcrue/negative_sampling.py','src/tcrue/vdjdb.py']:
        sources.append(identity('source',P,rel,'ORIGINAL_SAMPLER_SOURCE'))
    oldregistry=original['inputs']['positive_registry'];sources.append(identity('source',P,oldregistry['path'],'ORIGINAL_REGISTRY_FOR_3_SCOPE_SANITY_REPRODUCTIONS_ONLY',oldregistry['sha256']))
    posmanifest=original['inputs']['positive_manifest'];sources.append(identity('source',P,posmanifest['path'],'ORIGINAL_POSITIVE_QUOTA_MANIFEST',posmanifest['sha256']))
    pm=obj(P/posmanifest['path']);splitmanifest=pm['inputs']['positive_manifest'];sources.append(identity('source',P,splitmanifest['path'],'ORIGINAL_SPLIT_COMPONENT_MANIFEST',splitmanifest['sha256']))
    split=obj(P/splitmanifest['path']);sources.append(identity('source',P,split['config']['path'],'ORIGINAL_SPLIT_CONFIGURATION',split['config']['sha256']))
    assert original['resampling']['seed_by_resample']==dict(zip(['resample_'+str(i) for i in range(1,6)],[13,1013,2013,3013,4013]))
    tasks=[];seen=set()
    for r in sorted(graphs,key=lambda r:(r['train_or_test_draw_id'],r['fold_id'],['train','validation','test'].index(r['role']))):
        draw,fold,role=r['train_or_test_draw_id'],r['fold_id'],r['role']
        entry=original['folds'][fold]['partitions'][role];old=entry['resamples'][draw]
        assert old['negative_output']['path']==r['negative_path'] and old['negative_output']['sha256']==r['negative_sha256']
        assert entry['positive_input']['path']==r['positive_path'] and entry['positive_input']['sha256']==r['positive_sha256']
        seed=int.from_bytes(hashlib.sha256(f"{old['base_seed']}\0{fold}\0{role}".encode('ascii')).digest()[:8],'big')
        assert seed==old['derived_seed']==old['audit']['random_seed']
        for kind in ['positive','negative']:
            rel=r[kind+'_path']
            if rel not in seen:
                sources.append(identity('source',P,rel,'ORIGINAL_'+kind.upper()+'_IMMUTABLE_INPUT',r[kind+'_sha256']));seen.add(rel)
        tasks.append(dict(task_id=f'{draw}__{fold}__{role}',draw_id=draw,fold_id=fold,role=role,capacity_problem_id=r['capacity_problem_id'],positive_source_path=r['positive_path'],positive_sha256=r['positive_sha256'],positive_rows=entry['positive_input']['rows'],negative_source_path=r['negative_path'],old_negative_sha256=r['negative_sha256'],old_negative_rows=old['negative_output']['rows'],old_forbidden135555_keys=int(r['annotation_unique_keys']),old_unknown_keys=int(r['unknown_unique_keys']),base_seed=old['base_seed'],derived_seed=seed,action='REMATCH_ORIGINAL_SAMPLER' if r['conservative_policy_rematch']=='1' else 'BYTE_REUSE_AFTER_75_NEW_CHECKS',positive_output_path='inputs/'+r['positive_path'],negative_output_path='inputs/'+r['negative_path']))
    assert sum(r['action']=='REMATCH_ORIGINAL_SAMPLER' for r in tasks)==42
    cps=[]
    for r in sorted(capacities,key=lambda r:(r['fold_id'],['train','validation','test'].index(r['role']))):
        subset=[t for t in tasks if t['capacity_problem_id']==r['capacity_problem_id']]
        assert len(subset)==5 and len({t['positive_sha256'] for t in subset})==1 and r['all_five_draws_share_positive_identity']=='True'
        cps.append(dict(capacity_problem_id=r['capacity_problem_id'],fold_id=r['fold_id'],role=r['role'],positive_source_path=r['positive_path'],positive_sha256=r['positive_sha256'],positive_rows=subset[0]['positive_rows'],positive_output_path=subset[0]['positive_output_path'],draw_count=5,rematch_count=sum(t['action']=='REMATCH_ORIGINAL_SAMPLER' for t in subset)))
    writecsv(W/'task75.csv',tasks);writecsv(W/'capacity15.csv',cps)
    contract=dict(status='FROZEN_BEFORE_H1_ORIGINAL_REMATCH',created_utc=datetime.now(timezone.utc).isoformat(),scope='ONLY_ORIGINAL_R02_MAIN_25_MEMBERS_STRUCTURAL_INPUTS',generation='r05_original_repair_h1_20261009',authority_r05_manifest_sha256=R05_HASH,forbid_policy='CONSERVATIVE_EXACT_ELIGIBLE_ANNOTATION_EXCLUSION_135555',forbidden_registry_root='g2',forbidden_registry_path='registry/forbidden_registry.csv.gz',forbidden_registry_sha256=sha(G2/'registry/forbidden_registry.csv.gz'),forbidden_registry_keys=135555,unknown_semantics='Excluded conservatively; UNKNOWN is not a positive label.',source_root_server='/data/lwx/proj/TCR_unseen_epitope/workspaces/codex-tcrue-figure4-r',r05_root_server='/data/lwx/proj/TCR_unseen_epitope/runs/revision_20261008_v1_1/r05_decision_20261009',g2_root_server='/data/lwx/proj/TCR_unseen_epitope/runs/revision_20261008_v1_1/r04_structure_g2_20261009',science_image='sha256:d7c8838e67a156a9ed1abe981fee56f55a3637f2af47bae7b256a3c3c7551b59',sampler=original['matching'],base_seeds=original['resampling']['seed_by_resample'],seed_derivation='First 8 SHA256 bytes big endian of ASCII base_seed NUL fold_id NUL role; ORIGINAL namespace, no G2 namespace.',original_sampler_manifest_path=manifestrel,original_registry_path=oldregistry['path'],original_split_manifest_path=splitmanifest['path'],original_positive_quota_manifest_path=posmanifest['path'],preflight_order='fold_1..fold_5, train/validation/test; all15 complete graphs must PASS before any repair/reuse',rematch_required=42,rematch_by_role={'train':18,'validation':11,'test':13},conditional_byte_reuse_required=33,all_final_graph_audits_required=75,draw_repeat_policy='Retain and register repeated draws; no replacement seed or query deletion.',preserved=['exact original15 positive file bytes','original folds and peptide components','original per-query positive quotas','all75 original base/effective seeds','original sampler algorithm and negative IDs/CSV schema','33 clean old graph bytes'],prohibited=['G2 common-positive inputs','model operations','scores','performance','CI/P','reseed','delete query','split component','change positive quota','canonical comparator repair','parent mutation'],failure_policy='STOP immediately for infeasibility, changed42/33 action count, extra required input change, or original sampler reproduction failure; no partial release',sanity_reproduction_tasks=['resample_1__fold_1__train','resample_3__fold_3__validation','resample_5__fold_5__test'])
    writej(W/'contract.json',contract);writej(W/'authority_input_bindings.json',dict(status='PASS_SELECTED_HASH_BINDINGS_BEFORE_H1',files=sources))
    local=[]
    for name in ['prepare_h1_structure.py','run_h1_structure.py','launch_h1_structure.sh','contract.json','authority_input_bindings.json','task75.csv','capacity15.csv']:
        p=W/name;local.append(dict(path=name,role='PREMATCH_CODE_RULES_INPUT_MAP',sha256=sha(p),bytes=p.stat().st_size))
    writej(W/'code_rules_input_freeze.json',dict(status='FROZEN_BEFORE_ANY_H1_REMATCH',created_utc=datetime.now(timezone.utc).isoformat(),files=local,selected_input_files=sources))
    print(json.dumps(dict(status='PASS_H1_PREMATCH_FREEZE',tasks=75,capacities=15,rematches=42,conditional_byte_reuse=33,selected_sources=len(sources),freeze_sha256=sha(W/'code_rules_input_freeze.json'))))

if __name__=='__main__':main()
