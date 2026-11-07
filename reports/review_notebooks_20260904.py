"""Review notebook execution with MinIO writes redirected to memory.

Reads the configured project bucket; never changes remote objects or notebooks.
Run from the project root with its virtual-environment Python.
"""
import ast
import contextlib
import hashlib
import io
import json
import os
import time
import warnings
from pathlib import Path

os.environ['MPLBACKEND'] = 'Agg'
import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from botocore.config import Config
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
real = boto3.client('s3', endpoint_url=os.environ['MINIO_ENDPOINT'],
    aws_access_key_id=os.environ['MINIO_ACCESS_KEY'],
    aws_secret_access_key=os.environ['MINIO_SECRET_KEY'], region_name='us-east-1',
    config=Config(signature_version='s3v4', connect_timeout=5, read_timeout=10,
                  retries={'max_attempts': 1}))
bucket = os.environ['MINIO_BUCKET']
report = {'notebooks': [], 'checks': {}, 'live_objects': [], 'live_metadata': []}

def read_real(key):
    return pd.read_parquet(io.BytesIO(real.get_object(Bucket=bucket, Key=key)['Body'].read()), engine='fastparquet')

objects = [o for page in real.get_paginator('list_objects_v2').paginate(Bucket=bucket)
           for o in page.get('Contents', [])]
report['live_objects'] = [{'key': o['Key'], 'bytes': o['Size']} for o in objects]
manifest_keys = [o['Key'] for o in objects if o['Key'].startswith('metadata/etl_metadata/') and o['Key'].endswith('.parquet')]
live_metadata = pd.concat([read_real(k) for k in manifest_keys], ignore_index=True)
report['live_metadata'] = live_metadata.to_dict('records')
print('LIVE METADATA', live_metadata[['run_id', 'layer', 'dataset_name', 'row_count']].to_string(index=False), flush=True)

overlay = {}
class MemoryWrites:
    def put_object(self, *, Bucket, Key, Body, **kwargs):
        overlay[(Bucket, Key)] = bytes(Body)
        return {'ResponseMetadata': {'HTTPStatusCode': 200}}
    def get_object(self, *, Bucket, Key, **kwargs):
        if (Bucket, Key) in overlay:
            return {'Body': io.BytesIO(overlay[(Bucket, Key)])}
        return real.get_object(Bucket=Bucket, Key=Key, **kwargs)
    def get_paginator(self, operation):
        assert operation == 'list_objects_v2'
        class Paginator:
            def paginate(self, *, Bucket, Prefix='', **kwargs):
                keys = {o['Key'] for page in real.get_paginator(operation).paginate(Bucket=Bucket, Prefix=Prefix)
                        for o in page.get('Contents', [])}
                keys.update(k for b, k in overlay if b == Bucket and k.startswith(Prefix))
                yield {'Contents': [{'Key': k} for k in sorted(keys)]}
        return Paginator()

boto3.client = lambda *args, **kwargs: MemoryWrites()
plot_count = 0
def close_plot(*args, **kwargs):
    global plot_count
    plot_count += len(plt.get_fignums())
    plt.close('all')
plt.show = close_plot
notebooks = {}
states = {}
for path in sorted((ROOT / 'notebooks').glob('*.ipynb')):
    nb = json.loads(path.read_text(encoding='utf-8'))
    notebooks[path.name] = nb
    entry = {'name': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
             'cells': len(nb['cells']), 'code_cells': 0, 'unexecuted_saved_cells': [],
             'saved_errors': [], 'execution': [], 'warnings': []}
    state = {'__name__': '__main__'}
    started = time.monotonic()
    for i, cell in enumerate(nb['cells'], 1):
        if cell['cell_type'] != 'code':
            continue
        entry['code_cells'] += 1
        source = ''.join(cell['source'])
        if cell.get('execution_count') is None:
            entry['unexecuted_saved_cells'].append(i)
        for output in cell.get('outputs', []):
            if output.get('output_type') == 'error':
                entry['saved_errors'].append({'cell': i, 'name': output.get('ename')})
        ast.parse(source)
        try:
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                exec(compile(source, f'{path.name}:cell_{i}', 'exec'), state)
            entry['warnings'].extend(sorted(set(str(w.message) for w in caught)))
            entry['execution'].append({'cell': i, 'status': 'passed'})
        except Exception as exc:
            entry['execution'].append({'cell': i, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
            print('CELL FAILED', path.name, i, type(exc).__name__, str(exc), flush=True)
            break
    entry['seconds'] = round(time.monotonic() - started, 2)
    entry['warnings'] = sorted(set(entry['warnings']))
    states[path.name] = state
    report['notebooks'].append(entry)
    print('EXECUTED', path.name, 'seconds', entry['seconds'], flush=True)

raw = states['01_dataset_understanding_and_raw_ingestion.ipynb']
silver = states['02_raw_to_silver_layer.ipynb']
gold = states['03_silver_to_gold_layer.ipynb']
eda = states['04_eda_and_sensor_analysis.ipynb']
final = states['05_Feature engineering_Final.ipynb']
old = states['05_Feature engineering.ipynb']
checks = report['checks']
for kind, name in [('train', 'raw_df_train'), ('test', 'raw_df_test'), ('rul', 'raw_df_RUL')]:
    df = raw[name]
    checks[f'source_{kind}'] = {'shape': list(df.shape), 'engines': int(df.unit_number.nunique()),
                              'nulls': int(df.isna().sum().sum()),
                              'infinities': int(np.isinf(df.to_numpy()).sum())}

for kind, variable in [('train', 'train_df_minio'), ('test', 'test_df_minio')]:
    df = final[variable]
    complete = df[df.time_in_cycles >= 20]
    checks[f'features_{kind}'] = {'shape': list(df.shape), 'engines': int(df.unit_number.nunique()),
       'duplicate_keys': int(df.duplicated(['unit_number', 'time_in_cycles']).sum()),
       'unique_columns': bool(df.columns.is_unique), 'complete_window_rows': len(complete),
       'rows_with_null': int(df.isna().any(axis=1).sum()),
       'complete_window_nulls': int(complete.isna().sum().sum()),
       'infinities': int(np.isinf(df.to_numpy()).sum()),
       'null_counts': {c: int(n) for c, n in df.isna().sum().items() if n}}
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine='fastparquet')
    buffer.seek(0)
    pd.testing.assert_frame_equal(df, pd.read_parquet(buffer, engine='fastparquet'))
    checks[f'parquet_roundtrip_{kind}'] = True

train = final['train_df_minio']
test = final['test_df_minio']
assert train.drop(columns='remaining_useful_life').columns.equals(test.columns)
checks['train_test_feature_columns_match'] = True
expected_rul = train.groupby('unit_number').time_in_cycles.transform('max') - train.time_in_cycles
assert expected_rul.equals(train.remaining_useful_life)
assert (train.groupby('unit_number').tail(1).remaining_useful_life == 0).all()
checks['training_target_correct'] = True
pd.testing.assert_series_equal(gold['rul_training_features'].remaining_useful_life, train.remaining_useful_life)
checks['notebooks_03_and_05_targets_match'] = True
pd.testing.assert_frame_equal(old['train_df_minio'], train.drop(columns='remaining_useful_life'))
checks['old_and_final_training_features_match'] = True
checks['old_test_shape'] = list(old['test_df_minio'].shape)
terminal = eda['rul_and_test_df'].groupby('unit_number').tail(1).set_index('unit_number').remaining_useful_life.sort_index()
labels = raw['raw_df_RUL'].set_index('unit_number').remaining_useful_life.sort_index()
pd.testing.assert_series_equal(terminal, labels)
checks['eda_terminal_labels_match'] = True
checks['eda_train_heatmap_includes_target'] = (
    'remaining_useful_life' in eda['temp']
    and 'remaining_useful_life' in eda['corr_matrix'].columns
)
checks['test_shortest_history'] = int(test.groupby('unit_number').time_in_cycles.max().min())

# Independent temporal checks: future perturbations, engine isolation, and known linear slope.
prepare = final['prepare_feature_input']
engineer = final['add_time_series_features']
sensors = final['sensors']
small = final['original_train_df_minio'].query('unit_number <= 2 and time_in_cycles <= 40').copy()
features = engineer(prepare(small), sensors)
prefix = engineer(prepare(small[small.time_in_cycles <= 25]), sensors)
pd.testing.assert_frame_equal(features[features.time_in_cycles <= 25].reset_index(drop=True), prefix)
changed = small.copy()
changed.loc[changed.time_in_cycles > 25, 'sensor_measurement_2'] += 100000
changed_features = engineer(prepare(changed), sensors)
pd.testing.assert_frame_equal(features[features.time_in_cycles <= 25], changed_features[changed_features.time_in_cycles <= 25])
checks['prefix_invariance_and_future_perturbation'] = True
single = engineer(prepare(small[small.unit_number == 2]), sensors)
pd.testing.assert_frame_equal(features[features.unit_number == 2].reset_index(drop=True), single)
checks['engine_isolation'] = True
shuffled = engineer(prepare(small.sample(frac=1, random_state=42)), sensors)
pd.testing.assert_frame_equal(features, shuffled)
checks['input_sorting'] = True
synthetic = prepare(small[small.unit_number == 1]).copy()
synthetic['current_sensor_measurement_2'] = 3 * synthetic.time_in_cycles + 7
synthetic_features = engineer(synthetic, ['current_sensor_measurement_2'])
np.testing.assert_allclose(synthetic_features.loc[synthetic_features.time_in_cycles >= 20, 'sensor_measurement_2_sensor_slope_20'], 3)
checks['known_linear_slope'] = True
synthetic.loc[synthetic.time_in_cycles == 20, 'current_sensor_measurement_2'] = 0
zero = engineer(synthetic, ['current_sensor_measurement_2'])
assert np.isnan(zero.loc[zero.time_in_cycles == 21, 'sensor_measurement_2_%change']).all()
checks['zero_denominator_safety'] = True

# Execute real Silver validation cells against deliberately damaged in-memory inputs.
validation_results = {}
for case in ['clean', 'sensor_nan', 'sensor_inf', 'negative_engine_id', 'missing_sensor_column',
             'exact_duplicate', 'conflicting_duplicate', 'missing_cycle', 'negative_rul']:
    state = {'pd': pd, 'np': np, 'train_df_minio': raw['raw_df_train'].copy(),
             'test_df_minio': raw['raw_df_test'].copy(), 'rul_df_minio': raw['raw_df_RUL'].copy()}
    if case == 'sensor_nan': state['train_df_minio'].loc[0, 'sensor_measurement_2'] = np.nan
    if case == 'sensor_inf': state['test_df_minio'].loc[0, 'sensor_measurement_2'] = np.inf
    if case == 'negative_engine_id': state['train_df_minio'].loc[state['train_df_minio'].unit_number == 1, 'unit_number'] = -1
    if case == 'missing_sensor_column': state['train_df_minio'] = state['train_df_minio'].drop(columns='sensor_measurement_2')
    if case in ['exact_duplicate', 'conflicting_duplicate']:
        extra = state['train_df_minio'].iloc[[0]].copy()
        if case == 'conflicting_duplicate': extra['sensor_measurement_2'] += 1
        state['train_df_minio'] = pd.concat([state['train_df_minio'], extra], ignore_index=True)
    if case == 'missing_cycle': state['train_df_minio'] = state['train_df_minio'].drop(index=5)
    if case == 'negative_rul': state['rul_df_minio'].loc[0, 'remaining_useful_life'] = -1
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            validation_cells = notebooks['02_raw_to_silver_layer.ipynb']['cells']
            silver_heading = next(
                i for i, cell in enumerate(validation_cells)
                if ''.join(cell['source']).startswith('## Silver layer')
            )
            for cell in validation_cells[6:silver_heading]:
                if cell['cell_type'] == 'code': exec(''.join(cell['source']), state)
        validation_results[case] = {'result': 'accepted', 'output_train_rows': len(state['silver_train_df'])}
    except Exception as exc:
        validation_results[case] = {'result': 'rejected', 'reason': f'{type(exc).__name__}: {exc}'}
checks['silver_validation_probes'] = validation_results

# Re-run Gold 03 in memory after 05 to demonstrate metadata replacement behavior.
memory_client = MemoryWrites()
metadata_key = final['metadata_key']
before = pd.read_parquet(memory_client.get_object(Bucket=bucket, Key=metadata_key)['Body'], engine='fastparquet')
gold_write_cell = next(
    cell for cell in notebooks['03_silver_to_gold_layer.ipynb']['cells']
    if cell['cell_type'] == 'code'
    and 'etl_metadata_df = read_parquet_object' in ''.join(cell['source'])
)
with contextlib.redirect_stdout(io.StringIO()):
    exec(''.join(gold_write_cell['source']), gold)
after = pd.read_parquet(memory_client.get_object(Bucket=bucket, Key=metadata_key)['Body'], engine='fastparquet')
checks['gold_metadata_before_rerun_03'] = before.loc[before.layer == 'gold', 'dataset_name'].tolist()
checks['gold_metadata_after_rerun_03'] = after.loc[after.layer == 'gold', 'dataset_name'].tolist()

# Compare current live layer objects with local source and their recorded counts.
live_checks = []
for _, record in live_metadata.iterrows():
    df = read_real(record.object_key)
    item = {'layer': record.layer, 'dataset': record.dataset_name, 'run_id': record.run_id,
            'shape': list(df.shape), 'row_count_matches': len(df) == record.row_count}
    source_name = {'train_data': 'raw_df_train', 'test_data': 'raw_df_test', 'RUL_data': 'raw_df_RUL'}.get(record.dataset_name)
    if source_name and record.layer in ['raw', 'silver']:
        clean = df.drop(columns=['processing_timestamp', 'data_quality_status'], errors='ignore')
        try:
            pd.testing.assert_frame_equal(clean, raw[source_name])
            item['matches_local_source'] = True
        except AssertionError:
            item['matches_local_source'] = False
    if record.dataset_name in ['train_features', 'test_features']:
        expected = train if record.dataset_name == 'train_features' else test
        try:
            pd.testing.assert_frame_equal(df, expected)
            item['matches_recomputed_features'] = True
        except AssertionError:
            item['matches_recomputed_features'] = False
    live_checks.append(item)
checks['live_readback'] = live_checks
checks['plots_executed'] = plot_count
checks['remote_writes_performed'] = 0
checks['in_memory_objects_written'] = len(overlay)
(ROOT / 'reports' / 'notebook_review_evidence_20260904.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
print('CHECKS', json.dumps({k: v for k, v in checks.items() if not k.startswith('features_')}, indent=2, default=str), flush=True)
print('FEATURE SHAPES', train.shape, test.shape, flush=True)
