from pathlib import Path
import tempfile,shutil,json,subprocess,csv,hashlib,datetime
r=Path('/data/lwx/proj/TCR_unseen_epitope/runs/revision_20261008_v1_1/r04_training_g2_20261009')
g2=r.parent/'r04_structure_g2_20261009'
fixture=Path(tempfile.mkdtemp(prefix='r04_metric_missing105_20261009_'))
for d in ('authority','execution','additive','metrics'): (fixture/d).mkdir()
for rel in ('summarize_r04_g2.py','validate_r04_metrics_g2.py','metrics/calculation_contract.json','metrics/code_contract_manifest.json','metrics/preperformance_metric_selfcheck.json','authority/model_contract.json'):
 shutil.copyfile(r/rel,fixture/rel)
(fixture/'execution/fit_ledger.json').write_text('[]\n');(fixture/'additive/artifact_index.json').write_text('{"artifacts":[]}\n')
command=['docker','run','--rm','--network','none','--entrypoint','python','-v',str(fixture)+':/work:rw','-v',str(g2)+':/g2:ro','d7c8838e67a1','/work/summarize_r04_g2.py','--root','/work','--g2-root','/g2']
p=subprocess.run(command,capture_output=True,text=True)
(fixture/'fixture_stdout.log').write_text(p.stdout);(fixture/'fixture_stderr.log').write_text(p.stderr)
def rows(rel):
 with (fixture/rel).open(newline='') as f:return list(csv.DictReader(f))
checks={'expected_nonzero_exit':p.returncode==2}
if (fixture/'metrics/metric_run_summary.json').exists():
 summary=json.loads((fixture/'metrics/metric_run_summary.json').read_text());dr=rows('metrics/draw_oof_summary.csv');qr=rows('metrics/per_query_oof.csv');rr=rows('metrics/recipe_summary.csv');tr=rows('metrics/task_audit.csv')
 checks.update({'105_task_FAIL_preserved':len(tr)==105 and all(x['status']=='FAIL' for x in tr),'21_draw_INCOMPLETE':len(dr)==21 and all(x['status']=='INCOMPLETE' for x in dr),'7_recipe_INCOMPLETE':len(rr)==7 and all(x['status']=='INCOMPLETE' for x in rr),'3528_common_query_rows_preserved':len(qr)==3528 and all(x['status']=='INCOMPLETE' for x in qr),'no_partial_macro_as_complete':all(x['interaction_AP']=='' and x['interaction_AUROC']=='' for x in dr),'no_missing_interaction_replaced_by_half':all(x['interaction_AP']=='' and x['additive_AP']=='' for x in qr),'summary_INCOMPLETE':summary['status']=='INCOMPLETE' and summary['passing_tasks']==0})
obj={'schema_version':1,'status':'PASS' if all(checks.values()) else 'FAIL','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'performance_read':False,'model_fits':0,'fixture_type':'Synthetic missing all 105 models, predictions and analytic artifacts; real G2 source identities read-only','fixture_root':str(fixture),'fixture_retained':True,'producer_code_sha256':hashlib.sha256((fixture/'summarize_r04_g2.py').read_bytes()).hexdigest(),'checks':checks,'fixture_stdout':p.stdout,'fixture_stderr':p.stderr}
out=r/'metrics/preperformance_failure_propagation_check.json'
with out.open('x') as f:json.dump(obj,f,indent=2);f.write('\n')
print(json.dumps(obj,indent=2));raise SystemExit(0 if all(checks.values()) else 2)
