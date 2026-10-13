"""Clarify source-field scope and frozen eligibility wording, without changing values."""
from pathlib import Path
import csv, hashlib, json
import openpyxl

HERE=Path(__file__).resolve().parent
REV=HERE.parent
PROJECT=HERE.parents[3]
DATA=REV/'source_data'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p): return str(p.relative_to(PROJECT))
def load(p): return json.loads(p.read_text())
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

tables={t:load(HERE/f'Table_{t}.json') for t in ['1','S1','S2']}
b=tables['1']['blocks'][0]
c=b['rows'][8][0]
assert c['display'] in ['Publications','Selected-reference publications (historical)']
c.update(value='Selected-reference publications (historical)',display='Selected-reference publications (historical)',
         source_file=rel(Path(__file__)),source_sha256=sha(Path(__file__)),
         field_selector={'format':'TEXT','literal':'Selected-reference publications (historical)'},
         display_rule='Editorial clarification of the inherited selected-reference field; unchanged historical counts')
exclusion_labels=[
    'Original exclusion audit: Shared publication/reference identifiers',
    'Original exclusion audit: Shared Development peptides',
    'Original exclusion audit: Exact positive-interaction overlap',
    'Original exclusion audit: Maximum peptide similarity to VDJdb Development',
]
for index,label in zip(range(14,18),exclusion_labels):
    cell=b['rows'][index][0]
    assert cell['display']==label
    cell.update(source_file=rel(Path(__file__)),source_sha256=sha(Path(__file__)),
                field_selector={'format':'TEXT','literal':label},
                display_rule='Explicit editorial label, distinguishing original exclusion fields from expanded source associations')
extra=' Publication counts of 159 and 34 refer to the historical selected-reference field, not updated totals across all source associations or counts of independent biological samples.'
if extra not in b['notes']: b['notes']+=extra
old='Actual complete-graph maximum flow and node-degree equality were checked in the frozen release.'
new='Maximum flow over all legal partition-local candidate edges and node-degree equality were checked in the frozen release.'
tables['S2']['blocks'][0]['notes']=tables['S2']['blocks'][0]['notes'].replace(old,new)
vdjdb=PROJECT/'results/revision_20261008_v1_1/r05_original_repair_h1_20261009/authority/source_snapshot/src/tcrue/vdjdb.py'
assert 'CLASSICAL_FULL_RESOLUTION_HLA' in vdjdb.read_text()
clarification={'source_file':rel(vdjdb),'source_sha256':sha(vdjdb),
               'field_selector':{'format':'PYTHON_SYMBOL','symbol':'CLASSICAL_FULL_RESOLUTION_HLA'},
               'meaning':'Classical HLA-A/B/C identifiers with at least two fields; preserve the complete normalized identifier'}
for row_index in [8,17]:
    cell=tables['S1']['blocks'][0]['rows'][row_index][4]
    cell['display']=cell['display'].replace('classical full-resolution HLA-A/B/C','classical HLA-A/B/C with at least two-field resolution')
    cell['display_rule']='Clarify the frozen regex meaning; source criterion and counts retained'
    cell['clarification_source']=clarification
note=' Here the historical term full-resolution means at least two-field resolution for classical HLA-A/B/C; the complete normalized identifier is retained.'
if note not in tables['S1']['blocks'][0]['notes']:tables['S1']['blocks'][0]['notes']+=note

all_cells=load(DATA/'all_displayed_table_cells.json')
docx=load(HERE/'tables_for_docx.json')
wb=openpyxl.load_workbook(HERE/'Additional_file_2_Supplementary_Tables_S1-S4.xlsx')
for t,obj in tables.items():
    save(HERE/f'Table_{t}.json',obj)
    md=[]
    for block in obj['blocks']:
        md += ['## '+block['table_id']+'. '+block['title'],'',
               '|'+ '|'.join(block['headers'])+'|','|'+'|'.join(['---']*len(block['headers']))+'|']
        md += ['|'+'|'.join(str(c['display']).replace('|','\\|') for c in row)+'|' for row in block['rows']]
        md += ['',block['notes'],'']
    (HERE/f'Table_{t}.md').write_text('\n'.join(md))
    prov=load(DATA/f'Table_{t}_provenance.json')
    for record in prov['cells']:
        block=next(x for x in obj['blocks'] if x['section']==record['panel_or_section'])
        row=record['table_row_index_zero_based'];col=block['headers'].index(record['metric'])
        cell=block['rows'][row][col]
        record.update(plotted_or_printed_value=cell['display'],value_full_precision=cell['value'])
        for key in ['source_file','source_sha256','field_selector','display_rule','clarification_source']:
            if key in cell:record[key]=cell[key]
        for entry in all_cells['cells']:
            if (entry['figure_or_table_id'],entry['panel_or_section'],entry['table_row_index_zero_based'],entry['metric'])==(record['figure_or_table_id'],record['panel_or_section'],row,record['metric']):
                entry.update(record)
    save(DATA/f'Table_{t}_provenance.json',prov)
    # Existing CSV layout is section plus the union of columns.
    old_rows=list(csv.DictReader((DATA/f'Table_{t}.csv').open()))
    if t=='1': old_rows[8]['Characteristic']=c['display']
    if t=='S1':
        for index in [8,17]:old_rows[index]['Criterion']=obj['blocks'][0]['rows'][index][4]['display']
    with (DATA/f'Table_{t}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(old_rows[0]));w.writeheader();w.writerows(old_rows)
    manifest=load(HERE/f'Table_{t}_manifest.json')
    manifest.setdefault('editorial_clarification_sources',{})[rel(Path(__file__))]=sha(Path(__file__))
    if t=='S1':manifest['editorial_clarification_sources'][rel(vdjdb)]=sha(vdjdb)
    for output in manifest['outputs']:
        p=PROJECT/output['path'];output.update(bytes=p.stat().st_size,sha256=sha(p))
    save(HERE/f'Table_{t}_manifest.json',manifest)
    for i,entry in enumerate(docx['main_tables']):
        if entry['table_id']==obj['table_id']:docx['main_tables'][i]=obj['blocks'][0]
    for entry in docx['supplement_table_catalog']:
        if entry['table_id']==obj['table_id']:
            entry['notes']=next(x for x in obj['blocks'] if x['sheet']==entry['sheet'])['notes']
    for block in obj['blocks']:
        if not block.get('sheet'):continue
        ws=wb[block['sheet']]
        if t=='S1':
            for row in [8,17]:ws.cell(row+2,5).value=block['rows'][row][4]['display']
        for contents_row in wb['Contents'].iter_rows(min_row=2):
            if contents_row[1].value==block['sheet']:contents_row[3].value=block['notes']
save(DATA/'all_displayed_table_cells.json',all_cells)
save(HERE/'tables_for_docx.json',docx)
wb.save(HERE/'Additional_file_2_Supplementary_Tables_S1-S4.xlsx')
print('Table1 historical-field label; S1 HLA criterion; S2 legal-candidate note clarified. Numerical values unchanged.')
