import pandas as pd
import numpy as np
import os
import json
from pandas.api.types import is_numeric_dtype
pj = os.path.join
pd.options.mode.chained_assignment = None

from keys import self_explanatory_med_names, io_concept_ids, lab_concept_ids, outcome_concept_ids, physiologic_concept_ids, unit_conversions, io_concepts_use

data_path = '/home/deverett/data/local/raw_queries/query_2024-11-03/'
caseid_file = pj(data_path, 'caseids.json')
tables = {
            'meds': 'aims_intraopmedications.h5',
            'phys': 'aims_intraopphysiologic.h5',
            'preop': 'processed_preop.h5',
            'ios': 'aims_intraopinputoutputtotals.h5',
            'notes': 'processed_notes.h5',
          }

MED_BOLUS, MED_INFUSION = 1, 2
ENFORCE_WEIGHT = True
use_table_prefix = True
timeless_singular_allrows = True
final_columns_remove = ['labs_0',
                        'phys_0',
                        'outcomes_90205',
                        'labs_poc___blood_gas___sample_type',
                        'labs_poc___glucose_(unspecified_source)',
                        'labs_medication_level___digoxin',
                        'labs_formal_lab___ethanol_level,_serum/plasma',
                        'labs_formal_lab___free_t4',
                        'labs_formal_lab___hcg,_urine',
                        ]
column_rename = dict(
                mpog_case_id='case_id',
                mpog_CaseID='case_id',
                mpog_med_concept_id='med_id',
                aims_med_name='med_nm',
                mpog_dose_type_cd='med_typ',
                aims_med_uom='med_u',
                aims_io_uom='io_u',
                aims_med_dose='med_dose',
                aims_patient_dosing_weight='med_w',
                mpog_physiologic_concept_id='phys_id',
                mpog_io_concept_id='io_id',
                aims_value_text='phys_val_txt',
                aims_value_numeric='phys_val_num',
                aims_sex='sex',
                date_of_death_range_start='deathdate',
	        aims_io_total='io_total',	
                aims_lab_observation_dt='lab_dt',
                mpog_lab_concept_id='lab_id',
                aims_outcome_observation_dt='outcome_dt',
                mpog_outcome_concept_id='outcome_nm',
        )

datetime_fields = ['aims_value_observation_dt',
                   'aims_dose_start_dt',
                   'aims_dose_end_dt',
                   'deathdate',
                   'aims_note_observation_dt',
                   'lab_dt',
                   'outcome_dt',
                  ]

index_timestamp_field = {
                         'meds': 'aims_dose_start_dt',
                         'phys': 'aims_value_observation_dt',
                         'ios': None,
                         'notes': 'aims_note_observation_dt',
                        }
med_end_timestamp_field = 'aims_dose_end_dt'

feature_specs = {
                'meds': ['med_nm'],
                'phys': ['phys_id'],
                'ios': ['io_id'],
                'notes': ['aims_note_concept_desc'],
                }
value_specs = {
                'meds': ['med_dose'],
                'phys': ['phys_val_txt', 'phys_val_num'],
                'ios': ['io_total'],
                'notes': ['aims_note_concept_desc'],
                }
timeless_singular_features = {
                    'preop': ['surgery', 'weight', 'height', 'sex', 'age', 'race', 'deathdate'], 
                    }
post_featurized_renames = [ 
                    ['phys', 'ios',],
                    [physiologic_concept_ids, io_concept_ids, lab_concept_ids, outcome_concept_ids]
                          ]
infusions_that_are_boluses = [
                                'normosol',
                                'ns infusion',
                                'lr infusion',
                                'lr iv infusion',
                            ]
prefixes_case_start = ['meds_', 'phys_']
fields_case_start_ignore = []

def rename_columns(data):
    for _, table in data.items():
        table.rename(columns=column_rename, inplace=True)
    return data

def convert_dates(data):
    for name, table in data.items():
        for field in datetime_fields:
            if field in table.columns:
                table[field] = pd.to_datetime(table[field])
    return data

def rename_meds(data):
    def _rename_meds(value):
        value = value.lower()
        for semn in self_explanatory_med_names:
            if semn in value:
                value = semn
                break
        return value
    data['meds']['med_nm'] = data['meds']['med_nm'].apply(_rename_meds)
    return data

def reunit_meds(data):
    def _reunit_meds(row):
        unit = row.med_u.lower()
        conv = unit_conversions[unit]
        conv_factor = conv[0]

        row.med_dose = row.med_dose * conv_factor
        row.med_u = unit
        
        if '/' in unit:
            row.med_typ = MED_INFUSION
        else:
            row.med_typ = MED_BOLUS

        return row
    data['meds'] = data['meds'].apply(_reunit_meds, axis=1)
    return data

def reunit_ios(data):
    def _reunit_ios(row):
        unit = row.io_u.lower()
        conv = unit_conversions[unit]
        conv_factor = conv[0]

        row.io_total = row.io_total * conv_factor
        row.io_u = unit
        
        return row
    data['ios'] = data['ios'].apply(_reunit_ios, axis=1)
    return data

def filter_ios(data):
    iod = data['ios']
    iod = iod[iod['io_id'].isin(io_concepts_use)]
    data['ios'] = iod
    return data

def encode_sex(val):
    val = str(val).lower()
    if val in ['-1', '-1.0', 'nan']:
        return np.nan

    if val == 'm':
        val = 1
    elif val == 'f':
        val = 2
    else:
        val = 0
    return val

def encode_rhythm(val):
    val = str(val).lower()
    if val in ['-1', '-1.0', 'nan']:
        return np.nan

    if 'af' in val or 'aflut' in val:
        val = 2
    elif 'nsr' in val:
        val = 1
    else:
        val = 3
    return val

def encode_ventmode(val):
    val = str(val).lower()
    if val in ['-1', '-1.0', 'nan']:
        return np.nan

    if 'man' in val or 'spont' in val:
        val = 1
    elif 'vc' in val:
        val = 2
    elif 'psv' in val:
        val = 3
    elif 'pcv' in val or 'pc' in val:
        val = 4
    elif 'ippv' in val:
        val = 5
    elif 'simv' in val:
        val = 6
    elif 'cpap' in val:
        val = 7
    elif 'ac' in val:
        val = 8
    elif 'fg ext' in val:
        val = 9
    else:
        val = 0
    return val

def encode_bloodtype(val):
    val = str(val).lower().strip()
    if val in ['-1', '-1.0', 'nan']:
        return np.nan

    if 'o negative' in val:
        val = 1
    elif 'o positive' in val:
        val = 2
    elif 'a negative' in val:
        val = 3
    elif 'a positive' in val:
        val = 4
    elif 'b negative' in val:
        val = 5
    elif 'b positive' in val:
        val = 6
    elif 'ab negative' in val:
        val = 7
    elif 'ab positive' in val:
        val = 8
    else:
        val = 0

    return val

def encode_abscreen(val):
    val = str(val).lower().strip()
    if val in ['-1', '-1.0', 'nan']:
        return np.nan

    if 'neg' in val:
        val = 1
    elif 'neg' in val:
        val = 2
    else:
        val = 0

    return val

def convert_labs_to_numerical(val):
    val = str(val).lower().strip()
    val = val.replace('>', '')
    val = val.replace('<', '')
    try:
        val = float(val)
    except:
        val = np.nan
    return val

def fill_mins(table, t0):
    start = t0
    end = table.index[-1]

    table = table[table.index >= start]

    filled = pd.DataFrame(index=pd.date_range(start, end, freq='min'))
    table = table.join(filled, how='outer')
    return table

def process_infusions(table):
    infusion_columns = [c for c in table.columns if c.endswith('-infusion')]

    starts = table.index.values
    for inf_col in infusion_columns:
        vals = table[inf_col].values
        ends = table[f'{inf_col}_end_timestamp'].values
        new_vals = []
        current_val = np.nan
        current_val_endtime = np.nan
        for start, val, end in zip(starts, vals, ends):

            if (not np.isnan(current_val_endtime)) and start >= current_val_endtime:
                current_val = np.nan
                current_val_endtime = np.nan

            if not np.isnan(val):
                current_val = val
                current_val_endtime = end
            
            elif np.isnan(val) and (not np.isnan(current_val)):
                pass

            new_vals.append(current_val)
        
        table.drop(f'{inf_col}_end_timestamp', inplace=True, axis=1)
        new_vals = np.array(new_vals)
        new_vals[np.isnan(new_vals)] = 0

        weight = table.weight.dropna().unique()
        if len(weight) != 1:
            raise Exception(f'In process_infusions, weight had no/multiple values:\n{weight}\n\n{table}')
        weight = weight[0]
        unit = inf_col.split('___')[-1].split('-')[0]
        is_weight_based = unit_conversions[unit][1]
        if is_weight_based:
            new_vals *= weight

        table.drop(inf_col, inplace=True, axis=1)
        new_col_name = inf_col.replace('___'+(inf_col.split('___')[-1]), '')
        if new_col_name in table: 
            table[new_col_name] = table[new_col_name].fillna(0)
            table[new_col_name] += new_vals 
        else:
            table[new_col_name] = new_vals
     
    return table

def get_case_ids():
    if os.path.exists(caseid_file):
        with open(caseid_file, 'r') as f:
            return json.loads(f.read())

    all_case_ids = []
    care_about = ['phys']

    caseid_ids = ['mpog_case_id', 'mpog_CaseID']

    for name in care_about:
        filename = tables[name]
        with pd.HDFStore(pj(data_path, filename), 'r') as h:
            allnames = [tname for tname in h if tname.strip('/').startswith(name)]
            for cid in caseid_ids:
                if cid in h[allnames[0]]:
                    break
            case_ids = pd.concat([h.select(nm, columns=[cid]) for nm in allnames])
            all_case_ids.append(np.unique(case_ids.values.squeeze()).tolist())
    
    if len(care_about) > 1:
        use_ids = []
        for cid in all_case_ids[0]:
            if all([cid in all_case_ids[i] for i in range(1, len(all_case_ids))]):
                use_ids.append(cid)
        ret = sorted(use_ids)
    else:
        ret = sorted(all_case_ids[0])

    if not os.path.exists(caseid_file):
        with open(caseid_file, 'w') as f:
            f.write(json.dumps(ret))
    return ret

def process_case(case_id):
    data = {}
    caseid_ids = ['mpog_case_id', 'mpog_CaseID']
    for name, filename in tables.items():
        with pd.HDFStore(pj(data_path, filename), 'r') as h:
            allnames = [tname for tname in h if tname.strip('/').startswith(name)]
            for cid in caseid_ids:
                if cid in h.select(allnames[0], start=0, stop=1).columns:
                    break
            data[name] = pd.concat( [h.select(nm, where=f'{cid} == "{case_id}"') for nm in allnames] )

    data = rename_columns(data)
    data = convert_dates(data)
    data = rename_meds(data)
    data = reunit_meds(data)
    data = reunit_ios(data)
    data = filter_ios(data)

    phys = data['phys']
    cphys = phys
    cpdt = cphys[index_timestamp_field['phys']].dropna()
    monitoring_start = cpdt.values.min()
    monitoring_end = cpdt.values.max()

    feature_tables = []
    for table_name, table in data.items():

        if table_name not in feature_specs:
            continue

        if table_name == 'labs':
            tss = table[index_timestamp_field['labs']]
            table = table[(tss>=monitoring_start) & (tss<=monitoring_end)]

        def featurize_row(row):
            if use_table_prefix:
                ret = f'{table_name}_' + '_'.join([f'{row[s]}' for s in feature_specs[table_name]])
            else:
                ret = f'_'.join([f'{row[s]}' for s in feature_specs[table_name]])

            if table_name == 'meds' and row.med_typ == MED_INFUSION and\
                    (row['med_nm'] not in infusions_that_are_boluses):
                ret += f'___{row.med_u}-infusion'

            return ret
        feature_prefixes = table.apply(featurize_row, axis=1).values
        
        for feature_prefix, rows in table.groupby(feature_prefixes):
            
            if index_timestamp_field[table_name] is not None:
                timestamp = rows[index_timestamp_field[table_name]].values
            else:
                timestamp = [None] * len(rows)
            feature_table = pd.DataFrame()
            feature_table['timestamp'] = timestamp
            feature_table['case_id'] = case_id
            
            if value_specs[table_name] is True:
                vals = [1] * len(rows)
                
            else:
                valid_value_fields = []
                for vfield in value_specs[table_name]:
                    if rows[vfield].isna().all():
                        continue
                    valid_value_fields.append(vfield)

                if len(valid_value_fields) == 0:
                    continue 

                elif len(valid_value_fields) == 1:
                    vfield = valid_value_fields[0]
                    vals = rows[vfield].values

                elif len(valid_value_fields) > 1:
                    vals = []
                    for _,row in rows.iterrows():
                        subvals = [row[vf] for vf in valid_value_fields if not pd.isna(row[vf])]
                        assert len(subvals)==1, f'Multiple value fields specified in a single obs: {valid_value_fields}, {subvals}'
                        vals.append(subvals[0])

            feature_name = f'{feature_prefix}'
            feature_table[feature_name] = vals
            
            if table_name == 'meds' and feature_prefix.endswith('-infusion'):
                feature_table[f'{feature_prefix}_end_timestamp'] = rows[med_end_timestamp_field].values

            feature_table.dropna(subset=feature_name, inplace=True)
            feature_tables.append(feature_table)

    case_table = pd.concat(feature_tables, ignore_index=True)
    case_table.drop_duplicates(inplace=True)

    for table_name, concept_id_key in zip(*post_featurized_renames):
        if use_table_prefix:
            id_str = {f'{table_name}_{i}':f'{table_name}_'+v.lower().replace('-','_').replace(' ','_') for i,v in concept_id_key.items()}
        else:
            id_str = {str(i):v.lower().replace('-','_').replace(' ','_') for i,v in concept_id_key.items()}
        case_table.rename(columns=id_str, inplace=True)
    
    ts = case_table.timestamp.values
    maxx = case_table.timestamp.dropna().max()
    ts[case_table.timestamp.isnull()] = maxx
    case_table['timestamp'] = ts

    def merge_same_times(df):
        if len(df) == 1:
            return df.iloc[0]
    
        non_nans = (~df.drop(['case_id'] + [c for c in df.columns if c.endswith('end_timestamp')], axis=1).isna())
        assert np.all(non_nans.sum(axis=1) == 1)
        
        def unn1(x):
            if len(x.dropna()) == 0: return np.nan
            return x.dropna().unique()[0]
        return pd.Series({c:unn1(df[c]) for c in df.columns})

    case_table = case_table.groupby(case_table.timestamp, dropna=False).apply(merge_same_times, include_groups=False)
    case_table.drop(['case_id'], axis=1, inplace=True)
    case_table.sort_index(inplace=True)
    
    pcs_cols = [c for pref in prefixes_case_start for c in case_table.columns if c.startswith(pref)]
    pcs_cols = [p for p in pcs_cols if p not in fields_case_start_ignore]
    sub = case_table[pcs_cols]
    is_empty = ((sub.isna()) | (sub==0)).all(axis=1).values
    if is_empty[0] == True:
        first_non_empty = np.argmin(is_empty)
    else:
        first_non_empty = 0
    t0 = sub.index[first_non_empty]

    case_table = fill_mins(case_table, t0=t0)
    
    dif = pd.to_datetime(case_table.index.values) - case_table.index[0]
    mins = dif.seconds / 60
    case_table['minutes_elapsed'] = mins

    for table_name, fields in timeless_singular_features.items():
        grps = data[table_name].groupby('case_id')
        for field in fields:
            if case_id not in grps.groups:
                case_table[field] = np.nan
            elif case_id in grps.groups:
                cid = grps.get_group(case_id)
                assert len(cid) == 1
                cid = cid.iloc[0]
                val = cid[field]
                if field == 'deathdate':
                    val = (val - case_table.index[0]).days
                if timeless_singular_allrows:
                    case_table[field] = val
                elif not timeless_singular_allrows:
                    case_table[field] = [val] + [np.nan]*(len(case_table)-1)

    if ENFORCE_WEIGHT:
        weight = case_table.weight.dropna().unique()
        if len(weight) == 0:
            return 'Stopping processing because no weight is found.'
    
    case_table = process_infusions(case_table)
    for column in [c for c in case_table.columns if c.startswith('meds')]:
        case_table[column] = case_table[column].fillna(0)
    case_table.drop(final_columns_remove, axis=1, inplace=True, errors='ignore')

    one_hots =[
            ['labs_formal_lab___blood_bank_abo_rh_type_interpretation', encode_bloodtype],
            ['labs_formal_lab___blood_bank_antibody_screen_interpretation', encode_abscreen],
            ['phys_cardiac_rhythm', encode_rhythm],
            ['phys_ventilator_mode', encode_ventmode],
            ['sex', encode_sex],
            ]
    for cname, cfxn in one_hots:
        if cname in case_table.columns:
            case_table[cname] = case_table[cname].apply(cfxn).astype(float)

    for c in [c for c in case_table.columns if c.startswith('labs')]:
        if case_table[c].dtype in ['O', str]:
            case_table[c] = case_table[c].apply(convert_labs_to_numerical).astype(float)

    for column in case_table.columns:
        try:
            case_table[column] = pd.to_numeric(case_table[column])
        except:
            pass
    
    for column in case_table.columns:
        if is_numeric_dtype(case_table[column]):
            isnan = case_table[column].isna()
            case_table[f'{column}_flag'] = ~isnan

            orig = case_table[column]
            orig[isnan] = 0
            case_table[column] = orig

    for column, dtype in case_table.dtypes.items():
        if dtype in [float, np.float16, np.float32, np.float64]:
            case_table[column] = case_table[column].astype(np.float32)
        elif dtype in [int, np.int16, np.int32, np.int64]:
            case_table[column] = case_table[column].astype(np.int32)

    return case_table

def commonize_columns(case_data):
    columns = []
    for cid, case in case_data.items():
        both = columns + case.columns.tolist()
        columns = np.unique(both).tolist()

    for cid, case in case_data.items():
        case = case_data[cid]
        toadd = [col for col in columns if col not in case.columns]
        toadd = pd.DataFrame(columns=toadd)
        case = pd.concat([case, toadd], axis=1)
        case = case.reindex(sorted(case.columns), axis=1)
        case_data[cid] = case

    return case_data

def merge_cases(case_data):
    all_cases = []
    for cid, case in case_data.items():
        case['case_id'] = cid
        all_cases.append(case)
    return pd.concat(all_cases)

def exclude_missing(case_ids, data, tables_of_interest=['phys', 'meds', 'preop',]):
    keep = []
    keys = [k for k in data.keys() if k in tables_of_interest]
    ucases = {tname:data[tname].case_id.unique() for tname in keys}
    for cid in case_ids:
        if all([cid in uc for _,uc in ucases.items()]):
            keep.append(cid)
    return np.array(keep)
