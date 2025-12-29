import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
from pandas.api.types import is_numeric_dtype
from manual_curation import curate, curation_exclude
import multiprocessing as mp
import tables
import warnings
warnings.simplefilter(action='ignore', category=tables.exceptions.NaturalNameWarning)

in_path = 'data/processed.h5'
out_path = 'data/ic.h5'

with pd.HDFStore(in_path, 'r') as hfile:
    ex = hfile.select('data', start=0, stop=1)
    cids = hfile.select_column('data', 'case_id')
    case_ids = np.unique(cids)
cols = [col for col in ex.columns if is_numeric_dtype(ex[col]) and (col.startswith('phys_')) and (not col.endswith('flag'))]
exclude = [ 'phys_exhaled_monitor_calculated_mac_equivalent',
            'phys_isoflurane_exp_%',
            'phys_sevoflurane_exp_%',
            'phys_desflurane_exp_%',
            'phys_nitrous_exp_%',
            'phys_nitrous_insp_%',
        ]
cols = [c for c in cols if c not in exclude]
extra = ['weight']
proxies = [
            ['phys_bp_dias_non_invasive', 'phys_bp_dias_arterial_line_(invasive,_peripheral)'],
            ['phys_bp_sys_non_invasive', 'phys_bp_sys_arterial_line_(invasive,_peripheral)'],
            ['phys_bp_mean_non_invasive', 'phys_bp_mean_arterial_line_(invasive,_peripheral)'],
        ]
sensible_ranges = { 
        'phys_bp_dias_non_invasive':[20,200],
        'phys_bp_dias_arterial_line_(invasive,_peripheral)':[20,200],
        'phys_bp_sys_non_invasive':[40,300],
        'phys_bp_sys_arterial_line_(invasive,_peripheral)':[40, 300],
        'phys_bp_mean_non_invasive':[35,150],
        'phys_bp_mean_arterial_line_(invasive,_peripheral)':[35,150],
        'phys_spo2_%': [60.0, 100.0],
        'phys_spo2_pulse_rate': [25.0, 300.0],
        }
notes_keep = ['notes_Intubation',
              'notes_Extubation',
              'notes_LMA Inserted',
              'notes_LMA Removed']

def process_notes(case):
    ncols = [c for c in case.columns if c.startswith('notes_')]

    drop = [c for c in ncols if c not in notes_keep]
    case = case.drop(drop, axis=1)
    
    summary = np.zeros(len(case))
    for col_idx, col_name in enumerate(notes_keep):
        abbrev = col_name.replace('notes_','')
        vals = case[col_name].values
        is_present = vals == abbrev
        summary[is_present] = col_idx + 1
    case['airway'] = summary
    case = case.drop(notes_keep, axis=1)

    return case

def sensify(case):
    for col in cols:
        dat = case[col]
        flag = case[f'{col}_flag'].values
        dat[flag == False] = np.nan
        
        if col in sensible_ranges:
            sr0, sr1 = sensible_ranges[col]
            dat[dat.values < sr0] = np.nan
            dat[dat.values > sr1] = np.nan

        flag[np.isnan(dat)] = False
        case[col] = dat
        case[f'{col}_flag'] = flag
    return case

def fill(sub):
    swap = []
    for col in cols:
        dat = sub[col]
        flag = sub[f'{col}_flag'].values
        dat[flag == False] = np.nan

        need = flag==False
        for proxy_grp in proxies:
            if col in proxy_grp:
                for proxy in proxy_grp:
                    if proxy == col:
                        continue
                    proxy_dat = sub[proxy]
                    proxy_flag = sub[f'{proxy}_flag'].values
                    proxy_has = proxy_flag == True
                    success = need & proxy_has
                    dat[success] = proxy_dat[success]

        dat_filled = dat.ffill(limit=None)
        dat_filled = dat_filled.bfill(limit=None).values
        assert not np.any(np.isnan(dat_filled))
        sub[col] = dat_filled
        sub[f'{col}_flag'] = flag
        swap += [col, f'{col}_flag']
    return sub, swap

def impute_curate(idx, cid):
    with pd.HDFStore(in_path, 'r') as hfile:
        case = hfile.select('data', where=f"'case_id' == {cid}")
    
    if curation_exclude(case):
        return None
    
    case = process_notes(case)
    case = sensify(case)
    case, _ = fill(case)
    case = curate(case)
    return case

concat_chunk = 1000
last_idx = 0
final_cids = []
for i in range(0, len(case_ids), concat_chunk):
    print(f'{i} / {len(case_ids)}\tImpute/curating       \r', end='', flush=True)

    with mp.Pool(64) as pool:
        case_tables = pool.starmap(impute_curate, enumerate(case_ids[i:i+concat_chunk]))

    print(f'{i} / {len(case_ids)}\tProcessing         \r', end='', flush=True)
    out = pd.concat(case_tables, copy=False)

    final_cids += out.case_id.unique().tolist()
    final_cids = np.unique(final_cids).tolist()
    
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

    print(f'{i} / {len(case_ids)}\tSaving out tables          \r', end='', flush=True)
    with pd.HDFStore(out_path, mode='a', complib='blosc', complevel=5) as store:
        store.append('data', out, data_columns=data_columns, index=True, min_itemsize=min_itemsize)

    last_idx += len(out)
    
with pd.HDFStore(out_path, mode='a', complib='blosc', complevel=5) as store:
    store.append('unique_case_ids', pd.DataFrame(np.array(final_cids).astype(int), columns=['case_id']), data_columns=True)
