import pandas as pd
import numpy as np
import pickle, gzip, json
import matplotlib.pyplot as pl
from matplotlib.transforms import blended_transform_factory as blend
from or_foundational.data_inspection import visualize
from or_foundational.params import input_features, output_features, colors, meds, nicknames, meds_micrograms, keep_feat, max_horizon, meds_mcg, context, bolus_mins, color_darkening, infusion_mins, bolus_rounding, infusion_rounding, bolus_typicals
from or_foundational.params.functions import load_data, select_subset, parse_into_contexts_and_horizons, medication_to_bolus_index_points, fold_ids_to_case_ids, find_fold_of_cid, generate_crossval_preds, compute_naive_predictions
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def _round(value, resolution):
    # resolution = "to the nearest"
    return round(round(value / resolution) * resolution, 3)
rround = lambda x, r: np.round(x / r) * r

data_path = 'data/processed.h5'
FS = 10
test_features = ['phys_bp_mean_non_invasive']
other_meds_win = 3 # mins

def med_overlap(i0, i1):
    i1_set = set(i1)
    queries = i0[:, None] + np.arange(-other_meds_win, other_meds_win+1)
    overlap = [len(set(qu).intersection(i1_set)) for qu in queries]
    overlap = np.array(overlap, dtype=bool)
    return overlap

def compute_ch_means(med_name, strat, strat_name, strat_labs, result_dict, valid):

    if isinstance(strat, str) and strat == 'dose':
        index_pts, dose = medication_to_bolus_index_points(data, med_name,
                                                           other_valid=valid,
                                                           return_doses=True,
                                                           offset=0)
        strat_cut = pd.qcut(dose, 3, duplicates='drop', precision=5)
        strat_fill = strat_cut.codes
        strat_labs = strat_cut.categories
        strat = pd.Series(np.zeros(len(data))*np.nan, index=data.index)
        strat.iloc[index_pts] = strat_fill + 1 # the latter +1 is to make 0 a non-category

    uvals = np.unique(strat.values)
    uvals = uvals[~np.isnan(uvals)]

    for uval_idx, uval in enumerate(uvals):

        strat_criteria = (strat == uval).values
        inclusion = valid & strat_criteria

        index_pts, doses = medication_to_bolus_index_points(data, med_name, other_valid=inclusion, offset=0, return_doses=True)
        
        has_other_meds = [med_overlap(index_pts, allmed_indices_and_doses[med]) for med in meds if med != med_name]
        has_other_meds = np.logical_or.reduce(has_other_meds)
        no_other_meds = ~has_other_meds
        index_pts = index_pts[no_other_meds]

        if len(index_pts) == 0:
            continue

        index_pts += 1
        context_data, horizon_data = parse_into_contexts_and_horizons(data,
                                                                      index_pts,
                                                                      context_length=5,
                                                                      flatten_context=False,
                                                                      max_horizon=15,
                                                                      context_columns=input_features,
                                                                      horizon_columns=test_features)
    
        context_orig = context_data.copy()
        context_data = context_data[..., -1]
        
        horizon_info = horizon_data.mean(axis=0)
        context_info = pd.Series(np.nanmean(context_data, axis=0), index=input_features) # mean over examples
        N = len(context_data)

        result_dict[med_name][strat_name] += [[context_info, horizon_info, strat_labs, N, context_orig]]

data = load_data(data_path)
valid = select_subset(data,
                      require_context=True,
                      require_horizon=True,
                      exclude_peri_intubation=True,
                      gas_used_in_case=True,
                      return_context_and_horizon=False)
allmed_indices_and_doses = {med_name:medication_to_bolus_index_points(data, med_name, other_valid=valid, offset=0) for med_name in meds}

meds_to_use = ['meds_propofol',
               'meds_ephedrine',
               'meds_phenylephrine',
               'meds_fentanyl',
               ]

all_cut = data.case_len > -1
age_cut = pd.qcut(data.age, 3).cat
weight_cut = pd.qcut(data.weight, 3).cat
sex_cut = data.sex == 1
cuts = {
        'All': [all_cut, [None]],
        'Age': [age_cut.codes, age_cut.categories],
        'Weight': [weight_cut.codes, weight_cut.categories],
        'Sex': [sex_cut, ['M','F']],
        'Dose': ['dose', None],
        }

rdict = {}
for med_name in meds_to_use:
    rdict[med_name] = {}
    for idx, (cut_name, (cut, cut_labs)) in enumerate(cuts.items()):
        print(med_name, cut_name)
        rdict[med_name][cut_name] = []
        compute_ch_means(med_name, cut, cut_name, cut_labs, rdict, valid=valid)

result_context = pd.DataFrame(index=rdict['meds_propofol']['All'][0][0].index)
result_horizon = {tf:pd.DataFrame() for tf in test_features}
result_context_full = {}
result_context_full_averaged = {}
strats_tosave = {}
for med,mdict in rdict.items():
    for svar,sdict in mdict.items():
        for sval,dic in enumerate(sdict):
            dcon, dhor, strat_labs, N, con_full = dic
            name = f'{med}__{svar}__{sval}__n={N}'
            
            assert np.all(result_context.index == dcon.index)
            result_context[name] = dcon.values

            result_context_full[name] = con_full
            result_context_full_averaged[name] = np.nanmean(con_full, axis=0)
            
            for tf, h in zip(test_features, dhor):
                result_horizon[tf][name] = h

            strats_tosave[name] = strat_labs

full = dict(context=result_context, horizon=result_horizon, strats=strats_tosave, context_full=result_context_full_averaged)
with gzip.open('data/mean_bolus_data.pickle', 'wb', compresslevel=5) as f: 
    pickle.dump(full, f, protocol=-1)

mean_synth_cases = {}
for med,mdict in rdict.items():
    for svar,sdict in mdict.items():
        for sval,dic in enumerate(sdict):
            dcon, dhor, strat_labs, N, con_full = dic
            name = f'{med}__{svar}__{sval}'

            mean = np.nanmean(con_full, axis=0)
            mean = pd.DataFrame(mean.T, columns=input_features)

            mean_synth_cases[name] = (mean, dhor)

# final randomly generated and screened case selection for clinician test
i_to_use = [
    13695012,
    29382519,
    7258043,
    7290671,
    15686745,
    11268705,
    8016314,
    23017064,
    18344668,
    8742173,
    24894677,
    21627633,
    504725,
    7801336,
    23422041,
    3346522,
    27409436,
    7608298,
    8110747,
    27175331,
    13390135,
    12748791,
    24339384,
    900501,
    13903877,
    3606187,
    27627109,
    15282690,
    15720395,
    15798536,
    9043163,
    21381859,
    16759237,
    19967660,
    13227487,
    10257682,
    11012009,
    11300078,
    23410320,
    7985966,
    6595280,
    28042212,
    9128254,
    17668175,
    14997000,
    5516608,
    9476200,
    9054218,
    25740530,
    12883537,
    5839919,
    13307793,
    3899847,
    26144813,
    15572427,
    9677740,
    14950471,
    1504732,
    14897072,
    17795713,
    16964215,
    16199973,
    2588649,
    25491874,
    18989582,
    9655628,
    25540380,
    10262268,
    5904488,
    1235879,
    14005517,
    24209147,
    3505836,
    11721504,
    22996463,
    9562628,
    1102271,
    24642322,
    13797772,
    7197034,
    7836167,
    14664044,
    23090733,
    21254413,
    11190252,
    25655127,
    19783922,
    8632930,
    7929618,
    6225121,
    18467499,
    9865792,
    27913343,
    21596175,
    18310658,
    17033829,
    577725,
    17228535,
    22733074,
    3093622,
    7205035,
    28320107,
    6380043,
    15694827,
    27378975,
    21978714,
    12662876,
    15563036,
    4133311,
    17442329,
    17700678,
    6039796,
    22441143,
    1999897,
    7522179,
    2429766,
    15620798,
    24500818,
    28627404,
    11544229,
    24030996,
    16707501,
    23741957,
    25282501,
    24648014,
    2053911,
    10781434,
    21486301,
    19588291,
    13684207,
    8703275,
    11677549,
    23967935,
    7006358,
    13108386,
    13450321,
    17824470,
    9111012,
    4482837,
    6530276,
    26787482,
    6581373,
    12394083,
    6810727,
    29279767,
    6725548,
    7496515,
    18222977,
    9746484,
    23788536,
    24865161,
    23184540,
    9974451,
    21514734,
    9240064,
    24030317,
    19618193,
    17646488,
    16161407,
    29424932,
    13356321,
    16442083,
    7661593,
    16286994,
    26293511,
    14370882,
    18723098,
    14502194,
    17033744,
    6637876,
    29141929,
    29431052,
    13946628,
    1117855,
    16984423,
    7178615,
    10142535,
    23748951,
    2340151,
    25740853,
    1002731,
    3680057,
    14889923,
    25374487,
    15143529,
    23035495,
    16769963,
    1204,
    15316893,
    16897128,
    19479589,
    17509621,
    25596238,
    29304624,
    26791200,
    25414618,
    3276933,
    26099524,
    4567700,
    20172440,
    2163477,
    26188020,
    11006977,
    10007176,
    27643991,
    23513336,
    3549034,
    27212681,
    2909016,
    17143086,
    26450620,
    19770574,
]
idx_pts_to_use = np.array(i_to_use)-1 # because they were specified as moment AFTER bolus, but here we specify as moment OF bolus

model_path = 'models/med_model.pickle'
fold_path = 'kfolds/kfold.npz'
feat_idx_to_use = 0

context_data_to_use, horizon_data_to_use = parse_into_contexts_and_horizons(data,
                                                              idx_pts_to_use+1,
                                                              context_columns=input_features,
                                                              horizon_columns=test_features,
                                                              max_horizon=15)
context_data_to_use = context_data_to_use[:, keep_feat]
baseline_data_to_use, _ = parse_into_contexts_and_horizons(data,
                                                          idx_pts_to_use+1,
                                                          context_length=1,
                                                          context_columns=test_features,
                                                          horizon_columns=test_features,
                                                          max_horizon=15)

true_to_use = horizon_data_to_use[:, feat_idx_to_use, :]
baseline_to_use = baseline_data_to_use

with open('surg_renames.json', 'r') as f:
    srn = json.loads(f.read())
surg_nicknames = srn
surg_nicknames['Mean'] = 'A typical surgery under GA'
meds_to_show = meds

result_horizon = {}
result_context = {}
for idx, idx_pt in enumerate(idx_pts_to_use.tolist() + list(mean_synth_cases.items())):

    if isinstance(idx_pt, tuple):
        ex_name, (case, hor) = idx_pt
        case['sex'] = np.round(case.iloc[0].sex)
        case['surgery'] = 'mean'
        idx_pt = len(case)-1

        true = hor.squeeze()
        assert len(test_features) == 1
        bl = case.iloc[-1][test_features[0]]
        true = np.append(bl, true)

        result_horizon[ex_name] = true.tolist()
        result_context[ex_name] = case

        case['minutes_idx'] = np.arange(len(case))
        min_idx = case.minutes_idx.values[-1]
        do_downsample = False
        mins_context_show = 5
        round_meds = True
        is_meancase = True

    else:
        cid = data.iloc[idx_pt].case_id
        case = data[data.case_id == cid]
        ex_name = f'{idx_pt}'

        true = true_to_use[idx]
        bl = baseline_to_use[idx]
        true = np.append(bl, true)

        result_horizon[ex_name] = true.tolist()
        result_context[ex_name] = case
        min_idx = int(data.iloc[idx_pt].minutes_idx)
        do_downsample = False
        mins_context_show = 5
        round_meds = False
        is_meancase = False

    eligible_time = (case.minutes_idx.values > min_idx-mins_context_show) & (case.minutes_idx.values <= min_idx)
    eligible_time_tvals = case.minutes_idx.values[eligible_time]
    
    age = int(case.age.unique()[0])
    weight = int(round(case.weight.unique()[0]))
    sex = {1:'male', 2:'female'}[case.sex.unique()[0]]
    surg = case.surgery.unique()[0].capitalize().replace('[phi]','')
    surg = surg_nicknames.get(surg, surg)

    mbp = case.phys_bp_mean_non_invasive.values
    sbp = case.phys_bp_sys_non_invasive.values
    dbp = case.phys_bp_dias_non_invasive.values
    hr = case.phys_spo2_pulse_rate.values
    
    def downsample(x, d=3):
        x = np.array(x)
        y = np.zeros_like(x) * np.nan
        x = x[::-1]
        y[::d] = x[::d]
        y = y[::-1]
        return y
    if do_downsample:
        sbp = downsample(sbp)
        dbp = downsample(dbp)
        mbp = downsample(mbp)
        hr = downsample(hr)
    
    vs_t = eligible_time_tvals
    sbp = sbp[eligible_time]
    dbp = dbp[eligible_time]
    mbp = mbp[eligible_time]
    hr = hr[eligible_time]

    fig, axs = pl.subplots(3, 1, figsize=(7,6),
                           sharex=True,
                           gridspec_kw=dict(left=0.22, right=0.98, bottom=0.2, top=0.9))

    ax = axs[2]
    msz = 5
    scl = 1.0
    ax.plot(vs_t, hr, marker='o', color='green', lw=0, alpha=0.7, mew=0)
    ax.plot(vs_t, sbp+msz*scl, marker='_', color='orangered', lw=0)
    ax.plot(vs_t, dbp-msz*scl, marker='_', color='orangered', lw=0)
    ax.plot(vs_t, mbp, marker='x', color='red', lw=0)
    
    minn = np.nanmin(np.concatenate([sbp, dbp, mbp, hr]))
    minn = np.floor(minn / 10) * 10
    maxx = np.nanmax(np.concatenate([sbp, dbp, mbp, hr]))
    ax.set_ylim([minn-5, maxx+8])
    ax.set_yticks(np.arange(minn, maxx+9, 10))
    ax.set_yticklabels([f'{yt:0.0f}' if i%2==0 else '' for i,yt in enumerate(np.arange(minn, maxx+9, 10))])
    ax.grid(True, axis='y', lw=0.25)
    ax.plot([-0.25], [0.8], marker='x', color='red', clip_on=False, transform=ax.transAxes)
    ax.text(-0.23, 0.8, 'MAP', color='red', clip_on=False, transform=ax.transAxes, ha='left', va='center', fontsize=9)
    ax.plot([-0.25], [0.2], marker='o', color='green', clip_on=False, transform=ax.transAxes)
    ax.text(-0.23, 0.2, 'HR', color='green', clip_on=False, transform=ax.transAxes, ha='left', va='center', fontsize=9)
    ax.plot([-0.25], [0.5], marker='_', color='orangered', clip_on=False, transform=ax.transAxes)
    ax.text(-0.23, 0.5, 'SBP/DBP', color='orangered', clip_on=False, transform=ax.transAxes, ha='left', va='center', fontsize=9)
    
    ax = axs[0]
    cmts = meds_to_show
    shown_idx = 0

    for mi,mts in enumerate(cmts):
        did_show = False
                
        if mts in meds_mcg:
            med_unit = 'mcg'
        else:
            med_unit = 'mg'

        mcol = colors[mts]
        mcol = color_darkening.get(mcol, mcol)

        kw_bolus = dict(marker='o', color=mcol, lw=0, markersize=10, alpha=0.4, mew=0)
        kw_inf = dict(marker=9, color=mcol, lw=0, markersize=12, alpha=0.3, mew=0)

        m = case[mts].values[eligible_time]
        
        first_elig_idx = np.argmax(eligible_time)
        last_mval = case[mts].values[first_elig_idx-1] if first_elig_idx > 0 else 0

        infusion_dose = 0
        for idx, (tp, mval) in enumerate(zip(eligible_time_tvals, m)):

            bolus_dose = 0
            
            is_bolus_type1 = (mval - last_mval) >= bolus_mins.get(mts, 1e-10) 
            is_bolus_type2 = np.any(np.isclose(mval, bolus_typicals.get(mts, [])))
            is_bolus = is_bolus_type1 or is_bolus_type2
            if is_bolus:
                ax.plot(tp, shown_idx, **kw_bolus)
                did_show = True
                bolus_dose = mval - last_mval if is_bolus_type1 else mval
                
                bolus_dose_r = _round(bolus_dose, bolus_rounding.get(mts))
                if med_unit == 'mcg':
                    bolus_dose_r *= 1000
                dose_str = f'{bolus_dose_r:0.0f}'
                ax.text(tp, shown_idx+0.18, dose_str,
                        ha='center', va='center', fontsize=9,
                        weight='bold')
                ax.text(tp, shown_idx-0.18, f'{med_unit}', fontsize=6,
                        ha='center', va='center')
               
            test_inf_dose = (mval-bolus_dose) * 1000 / weight 
            if test_inf_dose >= infusion_mins.get(mts, 1e-8):
                ax.plot(tp, shown_idx, **kw_inf)
                did_show = True
                
                new_infusion_dose = (mval-bolus_dose) * 1000 / weight
                
                min_increment = infusion_rounding.get(mts, 0)/2
                if new_infusion_dose - infusion_dose >= min_increment:
                    infusion_dose = new_infusion_dose
                    inf_dose_r = _round(infusion_dose, infusion_rounding.get(mts, 0))
                    if med_unit == 'mg':
                        dose_str = f'{inf_dose_r:0.0f}'
                    elif med_unit == 'mcg':
                        dose_str = f'{inf_dose_r:0.2f}'
                    ax.text(tp, shown_idx+0.18, dose_str,
                            ha='center', va='center', fontsize=9,
                            weight='bold')
                    ax.text(tp, shown_idx-0.18, 'mcg/kg/min',
                            ha='center', va='center', fontsize=4,)
                            
            else:
                infusion_dose = 0

            last_mval = mval

        if did_show:
            ax.text(-0.04, shown_idx, mts.replace('meds_',''), fontsize=10, ha='right', va='center',
                    color=mcol, transform=blend(ax.transAxes, ax.transData)) # med name
            shown_idx += 1
    ax.set_ylim([-1, shown_idx])
    ax.set_yticks([])
    
    ax.plot([-0.27], [1.16], marker='o', color='k', clip_on=False, transform=ax.transAxes,
            lw=0, markersize=10, alpha=0.4, mew=0)
    ax.text(-0.25, 1.16, 'Bolus', color='grey', clip_on=False, transform=ax.transAxes, ha='left', va='center', fontsize=8)
    ax.plot([-0.28], [1.00], marker=9, color='k', clip_on=False, transform=ax.transAxes,
            lw=0, markersize=9, alpha=0.35, mew=0)
    ax.text(-0.25, 1.00, 'Infusion', color='grey', clip_on=False, transform=ax.transAxes, ha='left', va='center', fontsize=8)

    ax = axs[1]
    sevo = case['phys_sevoflurane_exp_%'].values[eligible_time]
    iso = case['phys_isoflurane_exp_%'].values[eligible_time]
    des = case['phys_desflurane_exp_%'].values[eligible_time]
    nitrous = case['phys_nitrous_exp_%'].values[eligible_time]

    # PMID 23414556
    mac_sevo = 2.0
    mac_iso = 1.15
    mac_des = 6.0
    mac_nitrous = 105
    
    mac2mac = lambda m: m * 10 ** (-0.00269 * (age - 40))
    mac_sevo, mac_iso, mac_des, mac_nitrous = map(mac2mac, [mac_sevo, mac_iso, mac_des, mac_nitrous])
    gas = sevo/mac_sevo + iso/mac_iso + des/mac_des 
    gas = rround(gas, 0.2)

    ax.plot(eligible_time_tvals, gas, color='dimgrey', lw=2)
    ax.set_ylim([-0.1, 1.1])
    ax.grid(True, axis='y', lw=0.2)
    ax.text(-0.09, 0.5, 'end tidal\nMAC (volatile)', color='dimgrey',
            fontsize=10, ha='right', va='center', transform=ax.transAxes)

    if ('airway_flag' in case.columns and np.any(case.airway_flag)) or (is_meancase):

        if is_meancase:
            intub = -1000
        else:
            intub = np.unique(case.ett_or_lma.values)[0]
        
        if intub < np.min(eligible_time_tvals):
            intub = np.min(eligible_time_tvals) - 2
            imark = None
            intub_str = 'Airway secured\nbefore shown data'
        elif intub > np.max(eligible_time_tvals) + max_horizon:
            intub = np.max(eligible_time_tvals) + max_horizon
            imark = None
            intub_str = 'Airway secured\nlater than end'
        else:
            intub_str = 'Airway secured'
            imark = '^'

        axs[2].plot(intub, -0.17, marker=imark, color='purple', alpha=0.5, markersize=11, clip_on=False, mew=0,
                    transform=blend(axs[2].transData, axs[2].transAxes))
        axs[2].text(intub, -0.3, intub_str, ha='center', va='center', fontsize=9, color='purple',
                    transform=blend(axs[2].transData, axs[2].transAxes))
        for ax in axs:
            if imark is not None:
                ax.axvline(intub, color='purple', ls=':', alpha=0.6, lw=0.9)

    for ax in axs:
        ax.axvline(min_idx, color='k', ls='-', lw=0.75, alpha=0.3, zorder=1)

    xtt = np.arange(0, min_idx+max_horizon)
    ax.set_xticks(xtt)
    ax.set_xticklabels([f'{x-min_idx:0.0f}' if (x-min_idx)%3==0 else '' for x in xtt])

    for ax in axs:
        ax.set_xlim([min_idx-mins_context_show-0.6+1, min_idx + 15])

    axs[0].set_title(f'{age}-year-old {sex}, {weight}kg\n{surg}', fontsize=11, pad=15, weight='bold')

    axs[2].text(0.95, -0.3, 'Minutes', fontsize=11, transform=axs[2].transAxes, ha='center', va='center')

    end = axs[2].get_xlim()[-1]
    axs[2].axvspan(min_idx, end, color='orange', alpha=0.5, lw=0)
    axs[2].text((min_idx+end)/2, 0.5, 'Predict the MAP\nin this region', fontsize=9,
                ha='center', va='center', weight='bold', color='darkorange',
                transform=blend(axs[2].transData, axs[2].transAxes))
    
    for ax in axs[0:2]:
        end = ax.get_xlim()[-1]
        ax.axvspan(min_idx, end, color='grey', alpha=0.2, lw=0)
        _, ypos = ax.transData.inverted().transform(ax.transAxes.transform([0, 0.5]))
        ax.text((min_idx+end)/2, ypos, 'Not shown', fontsize=9,
                    ha='center', va='center', weight='bold', color='dimgrey')

    for ax in axs:
        ax.tick_params(labelsize=10)
        ax.grid(True, lw=0.2, color='k', alpha=0.25, axis='x')

    fig.text(0.5, 0.06, 'q1-minute BPs are always shown here for context\n(but this does not necessarily mean an arterial line was placed for this case).\nPlease note: you are being asked to predict MAP *every 3 minutes* starting from time 0.',
            fontsize=9,
            style='oblique', ha='center', va='center',)

    fig.savefig(f'images/{ex_name}.jpg', dpi=350)
    pl.close(fig)
##
