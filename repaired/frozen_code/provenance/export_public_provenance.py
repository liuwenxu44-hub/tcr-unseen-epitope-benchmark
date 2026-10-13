#!/usr/bin/env python3
"""Project only reviewed provenance fields; never fit, score, or modify sources."""
from __future__ import annotations
import argparse, csv, gzip, hashlib, io, json, os, pathlib, shutil


def long_path(value):
    value = os.path.abspath(os.fspath(value))
    if os.name == 'nt' and not value.startswith('\\\\?\\'):
        value = ('\\\\?\\UNC\\' + value[2:]) if value.startswith('\\\\') else ('\\\\?\\' + value)
    return pathlib.Path(value)


def join(root, relative):
    rel = pathlib.PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Unsafe relative path: ' + relative)
    return root.joinpath(*rel.parts)


def digest(file):
    h = hashlib.sha256()
    with open(file, 'rb') as fh:
        for block in iter(lambda: fh.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(file):
    op = gzip.open if str(file).endswith('.gz') else open
    return op(file, 'rt', encoding='utf-8-sig', newline='')


def projected(row, rule):
    out = {key: row[key] for key in rule['output_fields'] if key in row}
    if rule['operation'] == 'project_and_extract_public_references':
        raw = json.loads(row['source_record_json'])
        out['publication_ids_json'] = json.dumps(raw.get('publication_ids', []), ensure_ascii=False, separators=(',', ':'))
        out['assay_ids_json'] = json.dumps(raw.get('assay_ids', []), ensure_ascii=False, separators=(',', ':'))
        for key in ('source_record_index_1based', 'source_csv_data_row_1based', 'archive_member'):
            value = raw.get(key, '')
            out[key] = '' if value is None else str(value)
    if set(out) != set(rule['output_fields']):
        raise ValueError('Missing or extra projected field')
    return out


def write_projected(source, output, rule):
    count = 0
    cells = hashlib.sha256()
    with read_csv(source) as inp, open(output, 'wb') as raw_out:
        reader = csv.DictReader(inp)
        if reader.fieldnames != rule['input_fields']:
            raise ValueError('Source schema differs from reviewed allowlist: ' + rule['source'])
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw_out, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding='utf-8', newline='') as txt:
                writer = csv.DictWriter(txt, fieldnames=rule['output_fields'], lineterminator='\n')
                writer.writeheader()
                for row in reader:
                    out = projected(row, rule)
                    writer.writerow(out)
                    cells.update(json.dumps([out[k] for k in rule['output_fields']], ensure_ascii=False, separators=(',', ':')).encode('utf-8') + b'\n')
                    count += 1
    return count, cells.hexdigest()


def verify_projection(source, output, rule):
    count = 0
    cells = hashlib.sha256()
    with read_csv(source) as inp, read_csv(output) as outp:
        src = csv.DictReader(inp)
        public = csv.DictReader(outp)
        if src.fieldnames != rule['input_fields'] or public.fieldnames != rule['output_fields']:
            raise ValueError('Projection header mismatch')
        while True:
            left, right = next(src, None), next(public, None)
            if left is None or right is None:
                if left is not None or right is not None:
                    raise ValueError('Row count changed')
                break
            expected = projected(left, rule)
            if right != expected:
                raise ValueError('Retained field or row order changed at row ' + str(count + 1))
            cells.update(json.dumps([right[k] for k in rule['output_fields']], ensure_ascii=False, separators=(',', ':')).encode('utf-8') + b'\n')
            count += 1
    return count, cells.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True, help='Read-only frozen candidate')
    ap.add_argument('--policy', type=pathlib.Path, default=pathlib.Path(__file__).with_name('DATA_FIELD_POLICY.json'))
    ap.add_argument('--output', required=True, help='New or empty output directory')
    args = ap.parse_args()
    source, output = long_path(args.source), long_path(args.output)
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Source and output must be separate directory trees')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be absent or empty; inspect and choose a new directory')
    policy = json.loads(args.policy.read_text(encoding='utf8'))
    manifest = json.loads((source/'DERIVED_FILES.json').read_text(encoding='utf8'))
    inventory = {e['path']: e for e in manifest['files']}
    checked = []
    for rule in policy['rules']:
        inp = join(source, rule['source'])
        expected = inventory[rule['source']]
        if inp.stat().st_size != expected['bytes'] or digest(inp) != expected['sha256']:
            raise ValueError('Frozen source identity mismatch: ' + rule['source'])
        with read_csv(inp) as f:
            if csv.DictReader(f).fieldnames != rule['input_fields']:
                raise ValueError('Unreviewed source schema: ' + rule['source'])
        checked.append((rule, inp, expected))
    output.mkdir(parents=True, exist_ok=True)
    receipt = {'schema_version': 1, 'operation': 'ROW_PRESERVING_PUBLIC_PROVENANCE_PROJECTION', 'scientific_recomputation': False, 'source_manifest_sha256': digest(source/'DERIVED_FILES.json'), 'files': []}
    for rule, inp, expected in checked:
        out = join(output, rule['output'])
        out.parent.mkdir(parents=True, exist_ok=True)
        if rule['operation'] == 'copy_bytes':
            shutil.copyfile(inp, out)
            with read_csv(inp) as f:
                count = sum(1 for _ in csv.DictReader(f))
            cells = None
            if digest(inp) != digest(out):
                raise ValueError('Byte copy failed')
        else:
            count, cells = write_projected(inp, out, rule)
            if (count, cells) != verify_projection(inp, out, rule):
                raise ValueError('Projection verification failed')
        if digest(inp) != expected['sha256']:
            raise ValueError('Source changed during read')
        receipt['files'].append({'source': rule['source'], 'source_sha256': expected['sha256'], 'path': rule['output'], 'sha256': digest(out), 'bytes': out.stat().st_size, 'row_count': count, 'retained_cells_sha256': cells, 'retained_cells_and_row_order_exact': True, 'operation': rule['operation'], 'omitted_fields': [k for k in rule['input_fields'] if k not in rule['output_fields']]})
    receipt['status'] = 'PASS'
    (output/'PUBLIC_EXPORT_RECEIPT.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'status': 'PASS', 'files': [{'path':x['path'], 'rows':x['row_count'], 'sha256':x['sha256']} for x in receipt['files']]}))

if __name__ == '__main__':
    main()
