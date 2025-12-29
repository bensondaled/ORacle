import pandas as pd
import numpy as np
from or_foundational.params import meds, gases, phys, demog, context, max_horizon, peri_intubation_period, bolus_mins, gas_mins, output_features, demog_extended, bolus_typicals, bolus_maxs
import pickle

def select_subset(data,
                  require_context=True,
                  require_horizon=True,
                  exclude_peri_intubation=False, # ETT only
                  exclude_peri_airway=False, # ETT and LMA
                  exclude_nonsensical_airway_indicators=False,
                  only_peri_intubation=False,
                  gas_used_in_case=False,
                  return_context_and_horizon=True):

    has_context = data.minutes_idx >= context

    has_horizon = data.minutes_idx <= data.case_len - max_horizon

    pre_intubation_restriction = (data.minutes_idx < data.intubation) &\
            (data.intubation - data.minutes_idx > peri_intubation_period[0])
    post_intubation_restriction = (data.minutes_idx >= data.intubation) &\
            (data.minutes_idx - data.intubation > peri_intubation_period[1])
    intubation_restriction = pre_intubation_restriction | post_intubation_restriction
    intubation_restriction = intubation_restriction | (np.isnan(data.intubation.values))
    
    pre_airway_restriction = (data.minutes_idx < data.ett_or_lma) &\
            (data.ett_or_lma - data.minutes_idx > peri_intubation_period[0])
    post_airway_restriction = (data.minutes_idx >= data.ett_or_lma) &\
            (data.minutes_idx - data.ett_or_lma > peri_intubation_period[1])
    airway_restriction = pre_intubation_restriction | post_intubation_restriction
    airway_restriction = airway_restriction | (np.isnan(data.ett_or_lma.values))

    if 'gas_used' in data.columns:
        gas_used = data.gas_used.values
    else:
        gas_used = determine_case_gas_usage(data)

    nonsensical_airway_restriction = (gas_used==True) & (data.time_to_airway == 100000.0)

    valid = np.ones(len(data), dtype=bool)
    if require_context:
        valid = valid & has_context
    if require_horizon:
        valid = valid & has_horizon
    if exclude_peri_intubation:
        valid = valid & intubation_restriction
    if exclude_peri_airway:
        valid = valid & airway_restriction
    if exclude_nonsensical_airway_indicators:
        valid = valid & (~nonsensical_airway_restriction)
    if only_peri_intubation:
        valid = valid & (~intubation_restriction)
    if gas_used_in_case:
        valid = valid & gas_used
    if isinstance(valid, (pd.DataFrame, pd.Series)):
        valid = valid.values
    
    if return_context_and_horizon:
        valid_idxs = np.arange(len(data))[valid]

        context_idxs = valid_idxs[:, None] + np.arange(-context, 0)
        horizon_idxs = valid_idxs[:, None] + np.arange(0, max_horizon)

        context_data = np.take(data.values, context_idxs, axis=0)
        horizon_data = np.take(data.values, horizon_idxs, axis=0)
        return valid, context_data, horizon_data

    return valid

def determine_case_gas_usage(data):

    row_had_gas = np.any(data[gases].values > np.array([gas_mins[g] for g in gases]), axis=1)
    cids_had_gas = np.unique(data.case_id.values[row_had_gas])
    cid_key = {cid:False for cid in np.unique(data.case_id.values)}
    for cid in cids_had_gas:
        cid_key[cid] = True
    gas_used = data.case_id.map(cid_key).values
    return gas_used

def infer_intubation(case):
    # Note this is only ETT. If you wanted LMA, you'd include airway 1 or 3
    if np.any(case.airway == 1):
        return case.minutes_idx.values[case.airway.values == 1][0]
    else:
        return find_vent_start(case)
def infer_ett_or_lma(case):
    if np.any(case.airway.isin([1,3])):
        return case.minutes_idx.values[case.airway.isin([1,3]).values][0]
    return np.nan

def find_vent_start(case):
    if 'phys_ventilator_mode' not in case.columns:
        return np.nan

    vm = case.phys_ventilator_mode.values
    is_ventilated = (vm>1) & (vm<9)
    is_ventilated = np.append(True, is_ventilated)
    is_ventilated = is_ventilated.astype(int)
    start_vent = np.diff(is_ventilated)
    if np.any(start_vent == 1):
        start = case.minutes_idx.values[np.where(start_vent == 1)[0][0]]
        if np.sum(is_ventilated[1:][case.minutes_idx<start]) < 3:
            return start

    return np.nan

def fold_ids_to_case_ids(fold_split_path, fold_ids):
    if isinstance(fold_ids, (int, float)):
        fold_ids = [fold_ids]

    with np.load(fold_split_path) as tt:
        fold_data = tt['fold_id']
        cid_data = tt['case_id']
    relevant_cases = cid_data[pd.Series(fold_data).isin(fold_ids).values]

    return relevant_cases

def load_data(fname, quick=False, case_ids=None, extra_fields=[]):
    keep = meds + gases + phys + demog_extended + ['minutes_elapsed', 'case_id', 'airway', 'surgery', 'phys_ventilator_mode'] + extra_fields

    with pd.HDFStore(fname, 'r') as hfile:
        if 'unique_case_ids' in hfile:
            ucids = hfile['unique_case_ids'].case_id.values
        else:
            ucids = np.unique(hfile['data'].case_id.values)

    if case_ids is None:
        where_str = None
    else:
        case_ids = np.array(case_ids).tolist()
        where_str = f"'case_id' in {case_ids}"
    
    # validate which columns will exist
    with pd.HDFStore(fname, 'r') as hfile:
        ex = hfile.select('data', start=0, stop=1)
    keep = [k for k in keep if k in ex.columns]

    with pd.HDFStore(fname, 'r') as hfile:
        if quick:
            chunks = [chunk for chunk in hfile.select('data', columns=keep, start=0, stop=1000000, chunksize=10000, where=where_str)]
        else:
            chunks = [chunk for chunk in hfile.select('data', columns=keep, chunksize=10000, where=where_str)]
    
    data = pd.concat(chunks)
    
    data = data.sort_values(['case_id', 'minutes_elapsed'])

    case_len = data.groupby('case_id').apply(len)
    data['case_len'] = data.case_id.map(case_len)
    
    min_idx = data.groupby('case_id').apply(lambda case: case.minutes_elapsed.min())
    data['minutes_idx'] = data.minutes_elapsed - data.case_id.map(min_idx)

    data['intubation'] = data.case_id.map(data.groupby('case_id').apply(infer_intubation))
    data['ett_or_lma'] = data.case_id.map(data.groupby('case_id').apply(infer_ett_or_lma))
    data['airway_flag'] = (data['minutes_idx'] == data['ett_or_lma']).astype(int)
    data['time_to_airway'] = (data['ett_or_lma'] - data['minutes_idx']).fillna(1e5)

    data['gas_used'] = determine_case_gas_usage(data)

    # process weight data
    if 'height' in data.columns:
        mean_height = data.height[data.height>100].values.mean()
        new_height = data.height.values
        new_height[new_height<=100] = mean_height
        data['height_imputed'] = new_height
        data['bmi'] = data.weight.values / (data.height_imputed.values/100) ** 2
        is_male = data.sex.values == 1
        base = 2.3 * (data.height_imputed.values * 0.394 - 60)
        base[is_male] += 50
        base[~is_male] += 45.5
        data['ibw'] = base
        data['abw'] = data.ibw.values + 0.4 * (data.weight.values - data.ibw.values)
    else:
        data['ibw'] = data.weight.values
        data['abw'] = data.weight.values
        data['bmi'] = data.weight.values / 1.626**2 # mean human height

    return data

def parse_into_contexts_and_horizons(data, index_pts=None, context_length=context, max_horizon=max_horizon, context_columns=None, horizon_columns=None, flatten_context=True):
    '''index_pts : int array indicating row indices in `data` from which context & horizon should be extracted. Crtucially, like in the `select_subset` function, a context for timepoint T is the n_context rows *preceding* time T, and its horizon is the n_horizon rows *STARTING WITH* and then following time T. 

    In other words, if 42 is in index_pts, then:
    its generated context will be [...37,38,39,40,41]
    and its generated horizon will be [42,43,44,45,46,47...]

    For convenience, context and horizon data are formatted differently by default (can toggle off using parameter flatten_context):
    - context data are given as n_samples x n_features, where all context timepoints are collapsed into features because that's how models will generally accept it
    - horizon data are given as n_samples x n_features x n_horizon_timepoints

    '''

    if index_pts is None:
        index_pts = np.arange(len(data))
    if context_columns is None:
        context_columns = data.columns
    if horizon_columns is None:
        horizon_columns = data.columns
    
    context_takes = index_pts[:, None] + np.arange(-context_length, 0)
    horizon_takes = index_pts[:, None] + np.arange(0, max_horizon)

    context_data = np.take(data[context_columns].values, context_takes, axis=0)
    context_data = np.transpose(context_data, [0,2,1]) # samples x features x timepoints
    if flatten_context:
        context_data = context_data.reshape([len(context_takes), -1]) # [sample0_feat0_time0, sample0_feat0_time1 ...]

    horizon_data = np.take(data[horizon_columns].values, horizon_takes, axis=0)
    horizon_data = np.transpose(horizon_data, [0,2,1]) # samples x features x timepoints

    return context_data, horizon_data

def medication_to_bolus_index_points(data, med_name, other_valid=None, return_doses=False, offset=1):
    ''' given data and a med of interest, extract all boluses according to standardized criteria, and find the index points for all boluses. Crucially, with the default offset=1, these index points are the minute *after* the bolus, by the standard definition that an index point is item #0 of the horizon. In other words, if this function returns index point 42, it implies that med_name had a bolus at 41.
        
        other_valid: any other criteria (bool array of len(data)) to be combined with the medication info when deciding what is an included index point. joined with med boluses using AND logic. will apply offset to it, so if other_valid is [False, False, True, False], and the boluses occur at [False, False, True, False], then offset=1 means bolus indices are [False, False, False, True] and other_valid is used as [False, False, False, True] and thus that True is included. In other words it uses the actual time of bolus as the alignment with other_valid
    '''

    med = data[med_name].values.copy()
    
    dose_data = np.append(0, np.diff(med))
    bolus_type1 = dose_data > bolus_mins.get(med_name, 1e-10)
    bolus_toohigh = (bolus_type1) & (dose_data > bolus_maxs.get(med_name, 1e15))
    bolus_type2 = np.any(np.isclose(med[:,None], bolus_typicals.get(med_name, [])), axis=1)
    bolus = (bolus_type1 | bolus_type2) & (~bolus_toohigh)
    dose_data[(~bolus_type1) & (bolus_type2)] = med[(~bolus_type1) & bolus_type2]

    include = np.roll(bolus, offset) # these are our index points, ie time 0 of horizon, one min after bolus
    
    if other_valid is not None:
        include = include & np.roll(other_valid, offset)

    index_pts = np.arange(len(data))[include]

    if return_doses:
        dose = dose_data[index_pts - offset]
        return index_pts, dose

    return index_pts

def find_fold_of_cid(cid, fold_path):
    with np.load(fold_path) as fp:
        fold_data = fp['fold_id']
        cid_data = fp['case_id']

    yes = fold_data[cid_data == cid]
    yes = np.unique(yes)
    if len(yes) != 1:
        raise Exception(f'Instead of one unique fold being found, these folds were found: {yes}')

    return yes[0]

def compute_naive_predictions(data, index_pts,
                              strats=['age', 'weight'],
                              n_cut=3,
                              pred_features=output_features,
                              subtract_baseline=True,
                              baseline_mean_context=False):
    '''
    given data and index points, generate the "naive" model prediction, which means
    the average horizon from those index_pts (standard convention used per other functions),
    optionally baseline subtracted, (where baseline means value directly before horizon),
    but stratified by variables of interest in the original data.
    this is to mimic an intuition like "i think what will happen is the average response for someone in this age and weight group"

    baseline_mean_context: if subtracting baseline, this determines whether to use the data of the unitary moment at the index_pt, vs the avg over the n_context pts before it, as the definition of the baseline
    '''
    clen = context if baseline_mean_context else 1
    baseline_data, horizon_data = parse_into_contexts_and_horizons(data,
                                                                  index_pts,
                                                                  context_length=clen,
                                                                  context_columns=pred_features,
                                                                  horizon_columns=pred_features)
    if baseline_mean_context:
        assert len(pred_features) == 1, 'Didnt yet implement using the mean context as the baseline for more than one feature'
        baseline_data = baseline_data.mean(axis=-1)[:, None] # still want to shape it like it could have additional features, for compatibiltiy with the rest of code
    

    cuts = [pd.qcut(data[st], n_cut) for st in strats]
    grping = data.groupby(cuts, observed=False)
    ugrps = list(grping.groups.keys())
    group_id = np.zeros(len(data)) * np.nan

    for gidx, grp in enumerate(ugrps):
        where = grping.indices[grp]
        group_id[where] = gidx
    group_id = group_id[index_pts].astype(int) # the stratification group of each index pt (eg bolus)

    naive = [None for _ in range(len(ugrps))]
    for gidx, grp in enumerate(ugrps):
        criteria = group_id == gidx
        grab = horizon_data[criteria, :, :]
        if subtract_baseline:
            grab = grab - baseline_data[criteria][...,None]
        naive[gidx] = grab.mean(axis=0)
    naive = np.array(naive)

    naives = np.take(naive, group_id, axis=0) 

    return naives
     
def generate_crossval_preds(model_path, fold_path, context_data, cids, pred_features=output_features):
    '''
    given some input data and the associated case ids, use the correct cross validation model for each one to make a fair model prediction
    '''

    if fold_path is not None:
        cids = cids.astype(int)

        key = pd.DataFrame()
        with np.load(fold_path) as fp:
            key['fold'] = fp['fold_id']
            key.index = fp['case_id']
        key = key['fold'].to_dict()
        folds = pd.Series(cids).map(key)
        ufolds = np.unique(folds)
        
        preds = np.empty([len(context_data), len(pred_features), max_horizon])
        for fold in ufolds:
            where = (folds == fold).values

            with open(model_path, 'rb') as f:
                xgb = pickle.load(f)[fold]

            for feat_idx, feat in enumerate(pred_features):
                models = xgb[feat]['xgb']
                pred = np.array([m.predict(context_data[where, :]) for m in models]).T # sample x horizon
                preds[where, feat_idx, :] = pred

        return preds

    elif fold_path is None:
        cids = cids.astype(int)
        fold = 0 # have to choose one, just choose first
        with open(model_path, 'rb') as f:
            xgb = pickle.load(f)[fold]

        preds = np.empty([len(context_data), len(pred_features), max_horizon])
        for feat_idx, feat in enumerate(pred_features):
            models = xgb[feat]['xgb']
            pred = np.array([m.predict(context_data[:, :]) for m in models]).T # sample x horizon
            preds[:, feat_idx, :] = pred
        
        return preds
