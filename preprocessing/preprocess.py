import pandas as pd
import numpy as np
import os
pj = os.path.join
from datetime import datetime
import multiprocessing as mp
from unidecode import unidecode
from preprocessing_util import process_case, get_case_ids
import warnings
import tables
from warnings import simplefilter
simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=tables.exceptions.NaturalNameWarning)

dtstr = datetime.now().strftime('%Y-%m-%d_%H-%M')
data_path = 'data/'
tmp_dir = pj(data_path, f'tmp-processed_{dtstr}')

if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)

case_ids = sorted(get_case_ids())
print(f'Processing {len(case_ids)} cases.')

cid_key = pd.DataFrame(case_ids, columns=['mpog_case_id'])
mpgrp_idxs = np.array_split(cid_key.index.values, 250)
mpgrp_cids = np.array_split(cid_key.mpog_case_id.values, 250)
zipped = zip(mpgrp_idxs, mpgrp_cids)

def _parallel_process_case(idxs, case_ids):
    for idx, case_id in zip(idxs, case_ids):
        try:
            out_path = pj(tmp_dir, f'{idx}.feather')
            err_path = pj(tmp_dir, f'{idx}.error')

            result = process_case(case_id)

            if isinstance(result, str):
                raise Exception(result)

            result['case_id'] = idx

            result.reset_index(inplace=True, drop=True)

            result.to_feather(out_path)
        except Exception as e:
            with open(err_path, 'w') as ef:
                ef.write(str(e))
    return True

with mp.Pool(64) as pool:
    pool.starmap(_parallel_process_case, zipped)

# find common columns across all cases
print(f'Finding common columns across cases.', flush=True)
fs = [f for f in os.listdir(tmp_dir) if f.endswith('.feather')]

def _get_column_names(idx, fname):
    path = pj(tmp_dir, fname)
    table = pd.read_feather(path)
    columns = table.columns.tolist()
    return columns
    
with mp.Pool(64) as pool:
    columns = pool.starmap(_get_column_names, enumerate(fs))
columns = sorted(list(set([c for clist in columns for c in clist])))

# merge cases into table
print(f'Loading and commonizing, then merging, cases.', flush=True)
tmp_files = [f for f in os.listdir(tmp_dir) if f.endswith('.feather')]

def _prep_case_table(idx, path):
    path = pj(tmp_dir, path)
    case_table = pd.read_feather(path)
    
    not_there = [column for column in columns if column not in case_table.columns]
    nt_med = [nt for nt in not_there if nt.startswith('meds')]
    nt_other = [nt for nt in not_there if not nt.startswith('meds')]

    for col in nt_med:
        if col.endswith('_flag'):
            case_table[col] = True
        else:
            case_table[col] = 0
    
    for col in nt_other:
        if col.endswith('_flag'):
            case_table[col] = False
        else:
            case_table[col] = 0.0

    case_table = case_table.reindex(sorted(case_table.columns), axis=1)

    return case_table

out_name = f'processed_{dtstr}'
out_path = pj(data_path, f'{out_name}.h5')
idx = 0
while os.path.exists(out_path):
    out_name = f'processed_{dtstr}_{idx}'
    out_path = pj(data_path, f'{out_name}.h5')
    idx += 1

concat_chunk = 5000
last_idx = 0
for i in range(0, len(tmp_files), concat_chunk):
    print(f'{i} / {len(tmp_files)}\tParallel prepping tables\r', end='', flush=True)

    with mp.Pool(64) as pool:
        case_tables = pool.starmap(_prep_case_table, enumerate(tmp_files[i:i+concat_chunk]))

    print(f'{i} / {len(tmp_files)}\tProcessing tables        \r', end='', flush=True)
    out = pd.concat(case_tables, copy=False)
    
    col_rename = {c:c.replace('/','_') for c in out.columns if '/' in c}
    out = out.rename(columns=col_rename)

    min_itemsize = {}
    for column in out:
        try:
            if column.startswith('notes_') and (not column.endswith('_flag')):
                raise
            out[column] = pd.to_numeric(out[column]).astype(np.float32)
        except:
            out[column] = out[column].astype(str)
            out[column] = out[column].fillna('')
            out[column] = out[column].astype('object')
            minsize = 1500 if column=='surgery' else 50
            min_itemsize[column] = minsize
    out.index = range(last_idx, last_idx + len(out))

    data_columns = ['case_id', 'minutes_elapsed']

    print(f'{i} / {len(tmp_files)}\tSaving out tables          \r', end='', flush=True)
    with pd.HDFStore(out_path, mode='a', complib='blosc', complevel=5) as store:
        store.append('data', out, data_columns=data_columns, index=True, min_itemsize=min_itemsize)

    last_idx += len(out)

# save case id key
key_path = pj(data_path, f'processed_{dtstr}_caseid_key.csv')
cid_key.to_csv(key_path)
