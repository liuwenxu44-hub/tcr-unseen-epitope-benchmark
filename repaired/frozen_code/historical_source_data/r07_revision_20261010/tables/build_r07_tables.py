#!/usr/bin/env python3
"""Source extraction and formatting only. Never load models or calculate metrics."""
from pathlib import Path
import csv, json, hashlib, re, datetime, math
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, Alignment, Border, Side
from openpyxl.worksheet.table import Table, TableStyleInfo

PROJECT = Path(__file__).resolve().parents[4]
REV = PROJECT / 'results/revision_20261008_v1_1'
OUT = Path(__file__).resolve().parent.parent
TABLES = OUT / 'tables'
SOURCES = OUT / 'source_data'
COMMIT = '487c143dc1ee0d361a125b7d6adb774eb700d8e1'
H1 = REV / 'r05_original_repair_h1_20261009'
G2 = REV / 'r04_training_g2_20261009'
G2S = REV / 'r04_structure_g2_20261009'
R06 = REV / 'r06_discussion_20261009'
TABLE_FAMILY = 'Additional_file_2_Supplementary_Tables_S1-S4.xlsx'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def relative(path):
    return str(path.relative_to(PROJECT))

def read_json(path):
    return json.loads(path.read_text())

def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def pointer(data, path):
    for key in path.strip('/').split('/'):
        key = key.replace('~1', '/').replace('~0', '~')
        data = data[int(key)] if isinstance(data, list) else data[key]
    return data

def clean(v):
    # Display-only typography. Full precision is retained in each cell object.
    if v is None or v == '':
        return '—'
    if isinstance(v, bool):
        return 'Yes' if v else 'No'
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return format(v, '.3e') if v and abs(v)<1e-8 else format(v, '.4f')
    s = str(v)
    if re.fullmatch(r'-?\d{1,3}(?:,\d{3})+', s):
        return s.replace(',', '')
    if re.fullmatch(r'-?\d+\.\d+(?:[eE][+-]?\d+)?', s):
        v=float(s)
        return format(v, '.3e') if v and abs(v)<1e-8 else format(v, '.4f')
    return s

CACHE = {}
def cell(path, selector, value=None, display=None, inherited=None, key=None):
    path = Path(path)
    if path not in CACHE:
        CACHE[path] = {'sha256':sha(path), 'data':read_csv(path) if path.suffix=='.csv' else read_json(path) if path.suffix=='.json' else path.read_text()}
    if value is None:
        if isinstance(selector, str):
            value = pointer(CACHE[path]['data'], selector)
            locator = {'format':'JSON', 'pointer':selector}
        else:
            idx, col = selector
            value = CACHE[path]['data'][idx][col]
            locator = {'format':'CSV','data_row_index_zero_based':idx,'column':col}
            if key: locator['key'] = key
    else:
        locator = selector if isinstance(selector,dict) else {'format':'TEXT','locator':selector}
    result = {'value':value,'display':clean(value) if display is None else display,
              'source_file':relative(path),'source_commit':COMMIT,
              'source_sha256':CACHE[path]['sha256'],'field_selector':locator,
              'display_rule':'Source precision retained; four decimal display for metrics; tiny residuals scientific notation; no thousands separators'}
    if inherited: result['inherited_primary_provenance'] = inherited
    return result

def edit(text, authority=None):
    # Editorial labels are not analysed values; retain their actual code source.
    return cell(Path(__file__), {'format':'TEXT','literal':text}, value=text,
                inherited=authority)

def jcell(path, ptr): return cell(path, ptr)
def ccell(path, idx, col, key=None): return cell(path, (idx,col), key=key)
def dump(path, obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

TABLE_LIST = []
def add_table(table_id, section, sheet, title, headers, rows, notes, evidence='DESCRIPTIVE_NOT_INFERENTIAL'):
    assert rows and all(len(r)==len(headers) for r in rows)
    result={'table_id':table_id,'section':section,'sheet':sheet,'title':title,
            'headers':headers,'rows':rows,'notes':notes,'evidence_status':evidence}
    TABLE_LIST.append(result)
    return result

def from_csv(table_id, section, sheet, title, path, columns, notes, predicate=None):
    data=read_csv(path); rows=[]
    for i,r in enumerate(data):
        if predicate is None or predicate(r): rows.append([ccell(path,i,col) for _,col in columns])
    return add_table(table_id,section,sheet,title,[h for h,_ in columns],rows,notes)

def build():
    freeze=OUT/'authority/journal_standard_freeze_20261010/journal_standard_freeze_manifest.json'
    assert freeze.exists(), 'Missing journal-standard freeze'
    # Root completed create+validate; bind the exact immutable freeze rather than infer compliance.
    plan=read_json(TABLES/'table_and_supplement_plan.json')
    plan['journal_writing_gate']='ROOT_APPROVED_VALIDATED_STANDARD_FREEZE'
    plan['journal_standard_freeze']={'path':relative(freeze),'sha256':sha(freeze)}
    amendment=OUT/'authority/R07_masterplan_amendment.json'
    if amendment.exists(): plan['masterplan_amendment']={'path':relative(amendment),'sha256':sha(amendment)}
    dump(TABLES/'table_and_supplement_plan.json',plan)

    primary=H1/'metrics/results/primary_summary.json'
    old=PROJECT/'results/final/tables/table_1/table_1_source_data_long.csv'
    old_rows=read_csv(old)
    characteristics=['Source database','Positive interactions before cap','Positive interactions in final core cohort',
        'Degree-matched negatives in final core cohort','Unique TCR CDR3β sequences','Unique peptide sequences',
        'pMHC queries','HLA alleles','Publications','Query cap','Peptide-distance constraint']
    rows=[]
    for characteristic in characteristics:
        matching=[(i,r) for i,r in enumerate(old_rows) if r['characteristic']==characteristic]
        cells=[edit(characteristic)]
        for scope,cohort,pidx in [('internal','VDJdb internal benchmark',0),('external','IEDB external validation cohort',1)]:
            idx,r=next((i,r) for i,r in matching if r['cohort']==cohort)
            map_fields={'Positive interactions in final core cohort':'positive_count','Degree-matched negatives in final core cohort':'negative_count','pMHC queries':'query_count'}
            if characteristic in map_fields: cells.append(jcell(primary,f'/{pidx}/{map_fields[characteristic]}'))
            else: cells.append(cell(old,(idx,'formatted_value' if characteristic in ['Query cap','Peptide-distance constraint'] else 'raw_value'),inherited={k:r[k] for k in ['source_file','source_commit','source_sha256','field_selector','derivation']}))
        rows.append(cells)
    rows.extend([
        [edit('Current evaluation'),edit('Original five-fold OOF; 25 cell mean'),edit('Same original cohort; 25-member probability ensemble')],
        [edit('Current analysis status'),edit('Post hoc repair; descriptive'),edit('Post hoc same-cohort repair; descriptive')],
        [edit('External model selection'),edit('Not applicable'),edit('Not used for grid or checkpoint selection')]])
    for characteristic in ['Shared publication/reference identifiers','Shared Development peptides','Exact positive-interaction overlap','Maximum peptide similarity to VDJdb Development']:
        idx,r=next((i,r) for i,r in enumerate(old_rows) if r['characteristic']==characteristic and r['cohort']=='IEDB external validation cohort')
        rows.append([edit('Original exclusion audit: '+characteristic),edit('—'),cell(old,(idx,'raw_value'),inherited={k:r[k] for k in ['source_file','source_sha256','field_selector']})])
    ledger=R06/'evidence/claim_value_ledger.json'; ld=read_json(ledger)
    ci=next(i for i,c in enumerate(ld['claims']) if c['claim_id']=='r05_source_independence_scope')
    rows.append([edit('Expanded shared direct-support publications'),jcell(ledger,f'/claims/{ci}/display_values/shared_direct_support_publications'),jcell(ledger,f'/claims/{ci}/display_values/shared_direct_support_publications')])
    add_table('Table 1','cohorts',None,'Benchmark cohorts and current evaluation scope',
        ['Characteristic','VDJdb internal','IEDB external'],rows,
        'A pMHC query is a (peptide, HLA) pair. TCR, peptide, HLA and publication counts describe the retained positive cohorts. Degree matching equates positive and putative-negative degrees at every TCR and complete pMHC-query node; it does not establish experimental non-binding. Original exclusion-token overlap and expanded supporting-publication overlap use different source definitions. Shared donor or sample identity remains unresolved. Current IEDB results are a post hoc repair of the same cohort, rather than a new blind test.')

    rows=[]
    for pidx,scope in enumerate(['Internal','External']):
        for metric,suffix,ref,dr in [('Macro AP','AP','finite_random_no_tie_AP','delta_finite_random_AP'),('Macro AUROC','AUROC','random_AUROC','delta_random_AUROC')]:
            rows.append([edit(scope),edit(metric),jcell(primary,f'/{pidx}/interaction_{suffix}'),jcell(primary,f'/{pidx}/additive_{suffix}'),jcell(primary,f'/{pidx}/{ref}'),jcell(primary,f'/{pidx}/delta_Additive_{suffix}'),jcell(primary,f'/{pidx}/{dr}'),jcell(primary,f'/{pidx}/query_count')])
    add_table('Table 2','H1_current_primary',None,'Descriptive results after repair of the original analysis',
        ['Scope','Metric','Interaction','Additive','Finite ranking reference','Δ vs Additive','Δ vs reference','Queries'],rows,
        'AP denotes threshold-group average precision, rather than trapezoidal PR area. Internal estimates average queries equally within each five-fold OOF cell, then average all 25 cells equally. External probabilities are averaged across the same 25 models before query-equal metrics are calculated. These estimators are kept separate. The Additive comparator is the admitted analytic zero solution. The finite no-ties ranking expectation uses each query’s actual positive and negative counts. Neither reference defines a valid scientific no-pairing null. Current analyses are NOT_INFERENTIAL: no inferential confidence intervals, permutation P values or multiplicity-adjusted P values are reported. The external endpoint was specified before the original scoring, but the present repair is post hoc.')

    # Preserve existing literature comparison claims, without creating or extending them.
    t3=PROJECT/'results/final/tables/table_3/table_3_published_benchmarks.csv'
    lit=PROJECT/'results/final/tables/table_3/table_3_literature_registry.csv'
    litrows=read_csv(lit); comp=read_csv(t3)
    groups=[('A','Data and evaluation',[('Year','Year'),('Main data','Main data source'),('Task','Evaluation task')]),
            ('B','Splits and negative controls',[('Held-out peptide','Peptide-held-out split'),('Distance-aware','Distance-aware split'),('Negatives','Negative construction control'),('Marginal control','Query / TCR marginal control')]),
            ('C','External evaluation and reproducibility',[('Publication-independent test','Publication-independent external test'),('Protocol freeze / blind scoring','Pre-scoring protocol freeze / blind scoring'),('Retrieval','Retrieval evaluation'),('Code / data','Reproducible code and data')])]
    for label,title,cols in groups:
        rows=[]
        for i,r in enumerate(comp):
            if litrows[i]['study_id']=='this_study':continue
            row=[ccell(t3,i,'Study / method')]
            row[0]['citation_key']={'moris_imrex':'old01','titan':'old02','nettcr2':'old03','grazioli_tchard':'old09','deng_benchmark':'old07','dens_negative_bias':'old10','epic_trace':'old06','tspred':'old12','drost_epytope':'old11','lu_assessment':'old08'}[litrows[i]['study_id']]
            row[0]['doi']=litrows[i]['doi']
            row[0]['display']+=' [@'+row[0]['citation_key']+']'
            row[0]['display_rule']='Original study label plus canonical citation key; final numbers resolved at manuscript integration'
            for _,col in cols:
                if litrows[i]['study_id']=='lu_assessment' and col=='Year':
                    cross=PROJECT/'results/final/tables/table_3/source_cache/lu_assessment_crossref.json'
                    raw=read_json(cross); prefix='/message' if 'message' in raw else ''
                    row.append(jcell(cross,prefix+'/published-print/date-parts/0/0'))
                else:row.append(ccell(t3,i,col))
            rows.append(row)
        add_table('Table 3',label,None,title,['Study / method']+[h for h,_ in cols],rows,
            'Existing comparison categories are retained from the frozen literature table. NR means not reported; Partial follows the source table’s limited-support category. The comparisons describe study designs and do not rank absolute model performance. Published-study identifiers and primary-source locations are supplied in the provenance record. The current study is described separately in Tables 1–2.')

    s1=PROJECT/'results/final/tables/table_S1/table_S1_source_data_long.csv'
    from_csv('Supplementary Table S1','filtering','S1_Filtering','Eligibility, source screening and retained positive records',s1,
        [('Cohort','cohort'),('Step','step'),('Phase','phase'),('Stage','stage'),('Criterion','operational_criterion'),('Records','raw_records_retained'),('Previous (%)','retention_from_previous_formatted'),('Input (%)','retention_from_input_formatted')],
        'Existing stage counts and retention percentages are transcribed without recalculation. The cap defines the analysis subset; it is not an eligibility filter. References required a standardized PMID or DOI. The original reference rule, query eligibility and cap amendment preceded original external scoring. Retained original positive files were unchanged in H1.')

    graphs=H1/'structure/final_graph_audits75.json'; rows=[]
    gcols=[('Resample','resample_id'),('Fold','fold_id'),('Partition','partition'),('Queries','flow_audit/unique_query_count'),('TCRs','flow_audit/unique_tcr_count'),('Positives','independent_audit/positive_count'),('Negatives','independent_audit/negative_count'),('Matched fraction','independent_audit/matching_rate'),('TCR exact','independent_audit/tcr_degree_exact'),('Query exact','independent_audit/query_degree_exact'),('Forbidden','independent_audit/forbidden_registry_intersections'),('Duplicate negatives','independent_audit/duplicate_negative_interactions'),('Positive SHA256','positive_file_sha256'),('Negative SHA256','negative_sha256')]
    for i,_ in enumerate(read_json(graphs)):rows.append([jcell(graphs,f'/{i}/{col}') for _,col in gcols])
    add_table('Supplementary Table S2','H1_graphs','S2_H1_Inputs75','All repaired original fold-partition negative graphs',[h for h,_ in gcols],rows,
        'Each row is one input graph, not an independent biological sample. Actual complete-graph maximum flow and node-degree equality were checked in the frozen release. Per-partition HLA counts were not supplied in these frozen summaries and are not newly calculated here.')
    from_csv('Supplementary Table S2','H1_capacity','S2_H1_Capacity15','Original positive-graph capacity checks',H1/'structure/capacity15_summary.csv',[(x,x) for x in ['fold_id','role','status','maximum_flow_value','required_edges','positive_tcr_count','positive_full_query_count','original_quotas_preserved','full_complete_graph_maxflow','positive_file_sha256']], 'Counts and capacities are existing structure-audit results.')
    for label,path,key in [('H1',H1/'structure/leakage25.json','cross_role_max_similarity'),('G2',G2S/'execution/leakage_checks.json','cross_partition_max_similarity')]:
        rows=[]
        cols=[('Resample' if label=='H1' else 'Recipe','resample_id' if label=='H1' else 'recipe_id'),('Fold','fold_id'),('Train–validation similarity',key+'/train__validation'),('Train–test similarity',key+'/train__test'),('Validation–test similarity',key+'/validation__test'),('TCR-disjoint claimed','TCR_disjointness_claimed')]
        for i,_ in enumerate(read_json(path)):rows.append([jcell(path,f'/{i}/{c}') for _,c in cols])
        add_table('Supplementary Table S2',label+'_leakage','S2_'+label+'_Leakage',label+' partition separation',[h for h,_ in cols],rows,'The benchmark separates peptide components, not TCR identities. TCR overlap is permitted and is retained in the frozen leakage source.')
    from_csv('Supplementary Table S2','G2_inputs','S2_G2_Inputs315','All G2 fold-partition-draw input audits',G2S/'execution/input_results.csv',[(x,x) for x in ['recipe_id','fold_id','partition','draw_id','positive_count','negative_count','action','status','tcr_degree_exact','query_degree_exact','forbidden_intersections','negative_duplicate_keys','full_graph_maxflow_checked','base_seed','derived_seed','new_negative_sha256','positive_tcr_degree_sha256','positive_query_degree_sha256']], 'G2 input graphs belong to a separate common-positive generation and are not substituted for H1.')
    allocation=G2/'authority/final_allocations.json'; rows=[]
    acols=[('Recipe','config/recipe_id'),('Threshold','config/threshold'),('Split seed','config/split_seed'),('Query weight','config/query_weight'),('Record weight','config/record_weight'),('Allocation mode','allocation_mode'),('Components','graph/component_count'),('Canonical group SHA256','identity/canonical_component_group_hash'),('Task-set SHA256','identity/task_set_hash')]
    for recipe in read_json(allocation):rows.append([jcell(allocation,f'/{recipe}/{c}') for _,c in acols])
    add_table('Supplementary Table S2','G2_recipes','S2_G2_Recipes7','Fixed G2 split recipes and identities',[h for h,_ in acols],rows,'All adopted G2 allocation modes in this release are seeded_greedy, including S3 and S4. Allocation weights are bucket-objective weights, not model sample weights. The sequence threshold is an operational separation rule, not a biological cross-reactivity boundary.')

    h1cells=H1/'metrics/results/cell_metrics.csv'
    hcols=[(x,x) for x in ['set_id','train_resample_id','test_resample_id','query_count','positive_count','negative_count','singleton_positive_query_count','interaction_AP','interaction_AUROC','additive_AP','additive_AUROC','finite_random_no_tie_AP','delta_Additive_AP','delta_Additive_AUROC','delta_finite_random_AP','delta_random_AUROC','evidence_status']]
    from_csv('Supplementary Table S3','H1_cells','S3_H1_Cells25','All original internal train–test resample cells',h1cells,hcols,'Five disjoint test folds form each OOF cell. Repeated cells share cohorts and are not independent replicates.',lambda r:r['set_id']=='full')
    from_csv('Supplementary Table S3','H1_mask_cells','S3_H1_MaskCells75','Internal cells after each fixed source mask',h1cells,hcols,'Fixed complete-query deletion changes coverage and may break TCR-degree matching. No models are retrained and no new scores are generated.',lambda r:r['set_id']!='full')
    masks=H1/'metrics/results/six_fixed_source_sensitivity_summaries.csv'
    sumcols=[(x,x) for x in ['scope','set_id','query_count','positive_count','negative_count','singleton_positive_query_count','cell_count','interaction_AP','interaction_AUROC','additive_AP','additive_AUROC','finite_random_no_tie_AP','delta_Additive_AP','delta_Additive_AUROC','delta_finite_random_AP','delta_random_AUROC','evidence_status']]
    from_csv('Supplementary Table S3','H1_mask_summaries','S3_H1_Masks3','Internal fixed joint-source deletion summaries',masks,sumcols,'All three fixed source-mapping definitions are retained; no mask is selected by its performance.',lambda r:r['scope']=='internal')
    cov=H1/'metrics/results/coverage_and_deletion_audit.csv'
    covcols=[(x,x) for x in read_csv(cov)[0]]
    from_csv('Supplementary Table S3','H1_coverage','S3_H1_Coverage100','Internal mask coverage and retained degree audit',cov,covcols,'Postdeletion degree balance is reported as frozen. Complete-query deletion does not re-establish the original analytic-admission theorem.',lambda r:r['scope']=='internal')
    for section,sheet,title,name in [('G2_recipes','S3_G2_Recipes7','Complete G2 recipe summaries','recipe_summary.csv'),('G2_draws','S3_G2_Draws21','Every fixed G2 OOF draw','draw_oof_summary.csv'),('G2_vsS0_draws','S3_G2_vsS0_Draws21','Every same-draw G2 contrast with S0','vs_s0_draw_summary.csv'),('G2_vsS0_recipes','S3_G2_vsS0_Recipes7','Complete G2 versus-S0 summaries','vs_s0_recipe_summary.csv'),('G2_task_audit','S3_G2_Tasks105','Every G2 model task and artifact identity','task_audit.csv')]:
        path=G2/'metrics'/name
        from_csv('Supplementary Table S3',section,sheet,title,path,[(x,x) for x in read_csv(path)[0]],'All S0–S6 recipes and their three fixed draws are retained. Existing SD uses ddof=0; minima, maxima and SD are descriptive spread, not confidence intervals. S0 remains the sensitivity reference. No inferential test or external grid selection is supplied.')

    # Historical correction disclosure is an audit trail, not evidence of current performance.
    impact=REV/'r01/observed_impact_resolution.json'
    canonical=PROJECT/'results/logistic_regression/development_distance_aware_5fold_rebalanced_query_balanced_degree_matched_cap100_additive_only/summary.json'
    rows=[]
    for i,m in enumerate(read_json(impact)['models']):
        if not m['id'].startswith('additive_only__canonical__'):continue
        fold=m['id'].rsplit('__',1)[1]
        checkpoint=REV/'r01/execution/checkpoints'/(m['id']+'.json')
        rows.append([jcell(checkpoint,'/fold_id'),jcell(impact,f'/models/{i}/id'),jcell(impact,f'/models/{i}/parent_sha256'),
            jcell(impact,f'/models/{i}/old_coefficients_exact_zero'),jcell(impact,f'/models/{i}/old_intercept_exact_zero'),
            *[jcell(canonical,f'/folds/{fold}/test_metrics/{metric}') for metric in ['pooled_auroc','pooled_auprc','macro_auroc','macro_auprc']],
            jcell(checkpoint,'/numerical_diagnostics/old_train_probability_max_abs_from_half'),
            jcell(checkpoint,'/analytic_artifact/partition_score_checks/test/probabilities_exact_half')])
    add_table('Supplementary Table S3','historical_canonical_Additive','S3_History_AdditiveFolds','Historical Additive fold values and exact-zero audit',
        ['Fold','Historical checkpoint ID','Parent SHA256','Old coefficients exact zero','Old intercept exact zero','Old pooled AUROC','Old pooled AP','Old Macro AUROC','Old Macro AP','Old train max |p−0.5|','Analytic test scores exact half'],rows,
        'Historical correction disclosure only. These canonical-reference values are the former S3D rows, not H1 resample-cell results. R01 bound each old checkpoint to its summary and audited the exact analytic replacement. Folds with tiny nonzero parameters could impose an arbitrary ranking despite probabilities near one half. Analytic scores are tied exactly; their balanced-query AP and AUROC are one half. The canonical reference is not a sixth H1 resample.',
        evidence='HISTORICAL_CORRECTION_DISCLOSURE_NOT_CURRENT_EVIDENCE')
    external_history=REV/'r02/external/external_metrics_old_corrected.csv'
    rows=[]
    ecols=['generation','query_count','query_level_macro_auprc','query_level_macro_auroc','pooled_auprc','pooled_auroc']
    for i,r in enumerate(read_csv(external_history)):
        if r['level']=='full' and r['method']=='additive_only':rows.append([ccell(external_history,i,col) for col in ecols])
    add_table('Supplementary Table S3','historical_external_Additive','S3_History_ExternalAdditive','Historical external Additive correction',[ 'Historical generation','Queries','Macro AP','Macro AUROC','Pooled AP','Pooled AUROC'],rows,
        'Historical old-graph correction disclosure only. The old external Additive ensemble was replaced with the admitted analytic constant in R02; the same analytic-comparator principle is independently applied in H1. The old external interaction scores, contrasts, intervals and P values are not promoted into the current H1 results.',
        evidence='HISTORICAL_CORRECTION_DISCLOSURE_NOT_CURRENT_EVIDENCE')

    extrows=[]
    for i,r in enumerate(read_json(primary)):
        if r['scope']=='external':extrows.append([jcell(primary,f'/{i}/{c}') for _,c in sumcols])
    for i,r in enumerate(read_csv(masks)):
        if r['scope']=='external':extrows.append([ccell(masks,i,c) for _,c in sumcols])
    add_table('Supplementary Table S4','H1_external','S4_H1_FullAndMasks','Repaired external full cohort and all fixed masks',[h for h,_ in sumcols],extrows,'The same original external ensemble is evaluated in each fixed retained-query subset. The repair is post hoc; historical blinded scoring does not make this a new blinded validation. No confidence intervals or P values are available for current H1 estimates.')
    from_csv('Supplementary Table S4','H1_external_coverage','S4_H1_Coverage4','External coverage and deletion degree audit',cov,covcols,'The singleton-positive count is reported because AP has a query-size-dependent finite-ranking expectation. The retained degree audit accompanies every mask.',lambda r:r['scope']=='external')

    # Write each manuscript family as JSON/Markdown/CSV plus a cell-level provenance crosswalk.
    all_provenance=[]
    for table_id in ['Table 1','Table 2','Table 3','Supplementary Table S1','Supplementary Table S2','Supplementary Table S3','Supplementary Table S4']:
        blocks=[x for x in TABLE_LIST if x['table_id']==table_id]
        slug=table_id.replace('Supplementary ','').replace(' ','_')
        dump(TABLES/(slug+'.json'),{'table_id':table_id,'blocks':blocks})
        md=[]; flat=[]; prov=[]
        for block in blocks:
            md.extend([f"## {table_id}. {block['title']}",'','|'+'|'.join(block['headers'])+'|','|'+'|'.join(['---']*len(block['headers']))+'|'])
            for row_index,row in enumerate(block['rows']):
                md.append('|'+'|'.join(str(c['display']).replace('|','\\|').replace('\n',' ') for c in row)+'|')
                flat.append({'section':block['section'],**{h:c['display'] for h,c in zip(block['headers'],row)}})
                for col,c in zip(block['headers'],row):
                    pr={'figure_or_table_id':table_id,'panel_or_section':block['section'],'sheet':block['sheet'] or '',
                        'table_row_index_zero_based':row_index,'metric':col,'plotted_or_printed_value':c['display'],
                        'value_full_precision':c['value'],'analysis_set':block['section'],'confirmatory_status':block['evidence_status'],
                        **{k:c[k] for k in ['source_file','source_commit','source_sha256','field_selector','display_rule']}}
                    if 'inherited_primary_provenance' in c:pr['inherited_primary_provenance']=c['inherited_primary_provenance']
                    prov.append(pr)
            md.extend(['',block['notes'],''])
        (TABLES/(slug+'.md')).write_text('\n'.join(md)+'\n')
        fields=list(dict.fromkeys(k for r in flat for k in r))
        with (SOURCES/(slug+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(flat)
        dump(SOURCES/(slug+'_provenance.json'),{'table_id':table_id,'operation':'SOURCE_EXTRACTION_AND_DECLARED_DISPLAY_ONLY','cells':prov})
        all_provenance.extend(prov)
        selected={c['source_file']:c['source_sha256'] for c in prov}
        dump(TABLES/(slug+'_manifest.json'),{'table_id':table_id,'status':'DRAFT_FOR_INTEGRATION_NOT_SUBMISSION_READY','selected_sources':selected,'outputs':[{'path':relative(q),'bytes':q.stat().st_size,'sha256':sha(q)} for q in [TABLES/(slug+'.json'),TABLES/(slug+'.md'),SOURCES/(slug+'.csv'),SOURCES/(slug+'_provenance.json')]]})
    dump(TABLES/'tables_for_docx.json',{'citation_system':'numeric_square','main_tables':[x for x in TABLE_LIST if not x['table_id'].startswith('Supplementary')],'supplement_table_catalog':[{'table_id':x['table_id'],'title':x['title'],'sheet':x['sheet'],'notes':x['notes']} for x in TABLE_LIST if x['sheet']], 'supplementary_workbook':TABLE_FAMILY})
    dump(SOURCES/'all_displayed_table_cells.json',{'operation':'NO_STATISTICAL_RECOMPUTATION','cells':all_provenance})

    wb=Workbook();wb.remove(wb.active)
    ws=wb.create_sheet('Contents');ws.append(['Table','Worksheet','Content','Interpretation'])
    supplement=[x for x in TABLE_LIST if x['sheet']]
    for block in supplement:ws.append([block['table_id'],block['sheet'],block['title'],block['notes']])
    for block in supplement:
        ws=wb.create_sheet(block['sheet']);assert len(ws.title)<=31
        ws.append(block['headers']+['Source files','Source SHA256','Row selectors'])
        for row in block['rows']:
            vals=[]
            for c in row:
                raw=c['value'];s=str(raw)
                # Hashes, namespaces and long seeds stay text; metric values remain numeric.
                if isinstance(raw,(bool,int,float)):v=raw
                elif re.fullmatch(r'-?\d+',s) and len(s.lstrip('-'))<15:v=int(s)
                elif re.fullmatch(r'-?\d+\.\d+(?:[eE][+-]?\d+)?',s):v=float(s)
                else:v=s
                vals.append(v)
            files=list(dict.fromkeys(c['source_file'] for c in row))
            hashes=list(dict.fromkeys(c['source_sha256'] for c in row))
            # Explicit selectors cover each cell in the row without changing the source data.
            selectors={h:c['field_selector'] for h,c in zip(block['headers'],row)}
            ws.append(vals+['\n'.join(files),'\n'.join(hashes),json.dumps(selectors,ensure_ascii=False)])
        ws.freeze_panes='E2';ws.auto_filter.ref=ws.dimensions
        for row in ws.iter_rows(min_row=2):
            for c in row:
                if isinstance(c.value,float):c.number_format='0.000E+00' if c.value and abs(c.value)<1e-8 else '0.0000'
        ws.sheet_view.zoomScale=90
    for ws in wb:
        ws.freeze_panes=ws.freeze_panes or 'A2'
        for c in ws[1]:c.font=Font(name='Calibri',size=11,bold=True);c.alignment=Alignment(vertical='top',wrap_text=True);c.border=Border(bottom=Side(style='thin',color='000000'))
        ws.row_dimensions[1].height=34
        for row in ws.iter_rows(min_row=2):
            for c in row:c.font=Font(name='Calibri',size=11);c.alignment=Alignment(vertical='top',wrap_text=False)
        for col in range(1,ws.max_column+1):
            header=str(ws.cell(1,col).value or '')
            width=16 if any(x in header.lower() for x in ['ap','auroc','count','positive','negative','query','queries']) else 23
            if 'criterion' in header.lower() or ws.title=='Contents':width=50 if col>2 else 26
            if any(x in header.lower() for x in ['sha256','selector','source files']):width=24
            ws.column_dimensions[ws.cell(1,col).column_letter].width=width
        ws.sheet_properties.pageSetUpPr.fitToPage=True
        ws.page_setup.orientation='portrait';ws.page_setup.paperSize=ws.PAPERSIZE_A4
        ws.page_setup.fitToWidth=1;ws.page_setup.fitToHeight=0
    wb.save(TABLES/TABLE_FAMILY)
    verify=load_workbook(TABLES/TABLE_FAMILY,data_only=False,read_only=True)
    checks=[]
    for block in supplement:
        ws=verify[block['sheet']]
        actual=list(ws.values)
        assert len(actual)==len(block['rows'])+1,(block['sheet'],len(actual),len(block['rows']))
        for ri,row in enumerate(block['rows'],start=1):
            for ci,c in enumerate(row):
                got=actual[ri][ci];raw=c['value']
                if isinstance(got,(int,float)) and not isinstance(got,bool):
                    assert math.isclose(float(got),float(raw),rel_tol=1e-14,abs_tol=1e-15),(block['sheet'],ri,ci,got,raw)
                else:assert ('' if got is None else str(got))==str(raw),(block['sheet'],ri,ci,got,raw)
        checks.append({'sheet':block['sheet'],'source_rows':len(block['rows']),'roundtrip':'PASS_ALL_CELLS','source_data_unchanged':True})
    dump(TABLES/'workbook_roundtrip_validation.json',{'status':'PASS_SOURCE_VALUE_ROUNDTRIP','workbook':TABLE_FAMILY,'sha256':sha(TABLES/TABLE_FAMILY),'sheets':checks,'limitation':'Spreadsheet visual inspection is separate; this is not journal submission readiness.'})
    dump(TABLES/'table3_integration_review_status.json',{'status':'AUTHOR_CELL_REVIEW_PENDING','ten_published_rows_retained':True,'existing_current_study_row_removed':True,'primary_cache_hash_check':'37/37 source caches match (references subagent)','DOI_identity_check':'10/10 metadata identities match (references subagent)','new_semantic_cell_audit':False,'author_verified':False,'Lu_year':'published-print 2026; direct frozen Crossref selector controls the Year cell','restriction':'Do not convert cache or identity checks into a claim of new comparison-cell verification.'})
    print('Built',len(TABLE_LIST),'table sections;',len(all_provenance),'displayed cells;',len(supplement),'supplement sheets')

if __name__=='__main__':build()
