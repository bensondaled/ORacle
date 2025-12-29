import pandas as pd
import numpy as np
import pickle
from or_foundational.params import input_features, output_features, colors, meds, nicknames, meds_micrograms, max_horizon, canonical_doses, canonical_weight, phys, meds_use
from or_foundational.params.functions import load_data, select_subset, parse_into_contexts_and_horizons, medication_to_bolus_index_points, fold_ids_to_case_ids

data_path = 'data/processed.h5'
FS = 10
ncut = 4

data = load_data(data_path)
valid = select_subset(data,
                      require_context=True,
                      require_horizon=True,
                      exclude_peri_airway=True,
                      exclude_nonsensical_airway_indicators=True,
                      return_context_and_horizon=False)

def pretty(ax):
    ax.tick_params(labelsize=FS, width=0.5,)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
def slfmt(sl, mult=1):
    if isinstance(sl, str):
        return sl
    elif isinstance(sl, pd.Interval):
        return f'{sl.left*mult:0.0f}-{sl.right*mult:0.0f}'
    elif sl is None:
        return ''
    else:
        return sl

def draw_panel(med_name, strat, strat_name, strat_labs, axs, ylabs=False, lw=1):

    if isinstance(strat, str) and strat_name.startswith('Dose'):
        index_pts, dose = medication_to_bolus_index_points(data, med_name,
                                                           other_valid=valid,
                                                           return_doses=True,
                                                           offset=0)
        dose /= data.iloc[index_pts].weight.values # convert to mg/kg
        strat_cut = pd.qcut(dose, ncut, duplicates='drop', precision=5)
        strat_fill = strat_cut.codes
        strat_labs = strat_cut.categories
        strat = pd.Series(np.zeros(len(data))*np.nan, index=data.index)
        strat.iloc[index_pts] = strat_fill + 1 # the latter +1 is to make 0 a non-category

    uvals = np.unique(strat.values)
    uvals = uvals[~np.isnan(uvals)]
    diffs = []
    diffs_mg = []
    diffs_mgkg = []
    diffs_mgkgi = []
    dosedat = []

    for uval_idx, uval in enumerate(uvals):
    
        # -- calculation begins here --
        strat_criteria = (strat == uval).values
        inclusion = valid & strat_criteria

        index_pts, doses = medication_to_bolus_index_points(data, med_name,
                                                            other_valid=inclusion,
                                                            offset=0,
                                                            return_doses=True)
        if len(index_pts) == 0:
            continue
        weights = data.iloc[index_pts].weight.values
        ibws = data.iloc[index_pts].ibw.values
        abws = data.iloc[index_pts].abw.values
        bmis = data.iloc[index_pts].bmi.values

        _, horizon_data = parse_into_contexts_and_horizons(data,
                                                           index_pts,
                                                           max_horizon=max_horizon+1,
                                                           context_columns=output_features,
                                                           horizon_columns=output_features + [med_name])
        
        horizon_data[:, :-1, :] -=  horizon_data[:, :-1, 0][..., None] # diff from time 0, except dose
        means = horizon_data.mean(axis=0) # output features x timepoints

        baseline = means[:, 0][:,None]
        rest = means[:, 1:]
        most_diff_i = np.argmax(np.abs(rest - baseline), axis=1)
        diff = rest[np.arange(rest.shape[0]), most_diff_i]
        diff[-1] = baseline.squeeze()[-1] # for the med dose we care about baseline not max diff value
        
        diff_abs = diff
        diff_mg = diff / doses.mean()
        diff_mgkg = diff / (doses/weights).mean()
        diff_mgkgi = diff / (doses/ibws).mean()

        diffs.append(diff_abs)
        diffs_mg.append(diff_mg)
        diffs_mgkg.append(diff_mgkg)
        diffs_mgkgi.append(diff_mgkgi)
        
        dd = [doses.mean(), 
              (doses/weights).mean(),
              (doses/ibws).mean(),
              (doses/abws).mean(),
              ]
        dosedat.append(dd)

        alpha = 0.4 + 0.6 *  uval_idx / len(uvals) if len(uvals)>1 else 1.0
        for of, mean, ax in zip(output_features + [med_name], means, axs):
            ax.plot(np.arange(len(mean)), mean,
                    color=colors.get(of, 'k'),
                    alpha=alpha,
                    lw=lw)
            pretty(ax)

            if ylabs:
                if of == med_name:
                    nick = of.replace('meds_','').capitalize()
                else:
                    nick = nicknames.get(of, of)
                ax.set_ylabel(nick,
                              fontsize=FS,
                              color=colors.get(of, 'k'),
                              ha='center',
                              va='center',
                              rotation=0)
                ax.get_yaxis().set_label_coords(-0.9,0.5)

            if ax is axs[0]:
                mult = 1000 if med_name in meds_micrograms and strat_name=='Dose' else 1
                mult *= canonical_weight if strat_name=='Dose' else 1
                ax.text(0.5, 1.03 + 0.18*uval_idx,
                        f'{slfmt(strat_labs[uval_idx], mult)}',
                        fontsize=FS-3,
                        color='k',
                        alpha=alpha,
                        ha='center', va='center',
                        transform=axs[0].transAxes)
    axs[0].text(0.5, 1.05 + 0.18*(uval_idx+1),
                f'{strat_name}',
                fontsize=FS,
                weight='bold',
                ha='center', va='center',
                transform=axs[0].transAxes)

    
    return diffs, diffs_mg, diffs_mgkg, diffs_mgkgi, dosedat, strat_labs


all_cut = data.case_len > -1
age_cut = pd.qcut(data.age, ncut).cat
weight_cut = pd.qcut(data.weight, ncut).cat
sex_cut = data.sex == 1
cuts = {
        'All': [all_cut, [None]],
        'Age': [age_cut.codes, age_cut.categories],
        'Weight': [weight_cut.codes, weight_cut.categories],
        'Sex': [sex_cut, ['M','F']],
        'Dose': ['dose', None],
        }

dose_cuts = {}
diff_abs_data = {med:{} for med in meds_use}
diff_permg_data = {med:{} for med in meds_use}
diff_permgkg_data = {med:{} for med in meds_use}
diff_permgkgibw_data = {med:{} for med in meds_use}
dose_data = {med:{} for med in meds_use}

for med in meds_use:
    fig_mean, axs_mean = pl.subplots(len(output_features)+1, len(cuts),
                                     figsize=(7,7),
                                     gridspec_kw=dict(left=0.2, right=0.98, bottom=0.08, wspace=0.5,),
                                     sharey='row', sharex=True)

    for idx, (cut_name, (cut, cut_labs)) in enumerate(cuts.items()):
        lw = 2 if idx==0 else 0.9

        diffs, diffs_mg, diffs_mgkg, diffs_mgkgi, doses, sl = draw_panel(med, cut, cut_name, cut_labs,
                                                                       axs=axs_mean[:, idx],
                                                                       ylabs=idx==0,
                                                                       lw=lw)
        diff_abs_data[med][cut_name] = diffs
        diff_permg_data[med][cut_name] = diffs_mg
        diff_permgkg_data[med][cut_name] = diffs_mgkg
        diff_permgkgibw_data[med][cut_name] = diffs_mgkgi
        dose_data[med][cut_name] = doses
        dose_cuts[med] = sl
    
    for ax in axs_mean[-3]:
        ax.set_ylim([-0.5, 0.5]) # spo2
    for ax in axs_mean[-2]:
        ax.set_ylim([-4, 4]) # etco2

    fig_mean.text(0.5, 0.02, 'Minutes from medication',
             fontsize=FS,
             ha='center', va='center',
             transform=fig_mean.transFigure)
    fig_mean.savefig(f'/Users/bdd/Desktop/{med}.pdf')
    pl.close(fig_mean)

# peak effect summaries
def parse_cutlabs(labs, sname, mult=None):
    if sname == 'Sex':
        return ['M', 'F']
    elif mult is not None:
        return [f'{mult * l.left:0.0f}-{mult * l.right:0.0f}' for l in labs]
        #return [f'{mult * (l.left + l.right)/2:0.0f}' for l in labs]
    else:
        return [f'{l.left:0.0f}-{l.right:0.0f}' for l in labs]
def dose_str_labs(n):
    return ['Lowest'] + ['']*(n-2) + ['Highest']
def ylabel(ax, s):
    ax.text(-0.6, 0.5,
            s,
            ha='center',
            va='center',
            transform=ax.transAxes,
            fontsize=FS)

pvars_todo = ['phys_bp_mean_non_invasive', 'phys_spo2_pulse_rate', 'dose']
strats_todo = ['Age', 'Sex', 'Weight', 'Dose']
strat_units = ['(years)', '', '(kg)', '(mg/kg)']

for med in meds_use:
    unit_mult = 1000 if med in meds_micrograms else 1
    unit = 'mcg' if med in meds_micrograms else 'mg'
        
    fig, axs = pl.subplots(len(pvars_todo), len(strats_todo),
                           figsize=(7, 5),
                           gridspec_kw=dict(wspace=0.7, hspace=0.4, left=0.15, right=0.98,
                                            top=0.96, bottom=0.2),
                           sharex='col',
                           sharey='row',
                           )

    for pvi, pvar in enumerate(pvars_todo):
        pname = nicknames.get(pvar, pvar)
        pidx = (phys+['dose']).index(pvar)

        for sidx, strat in enumerate(strats_todo):
            ax = axs[pvi, sidx]

            dif_abs = np.array(diff_abs_data[med][strat])[:, pidx]
            dif_mg = np.array(diff_permg_data[med][strat])[:, pidx]
            dif_mgkg = np.array(diff_permgkg_data[med][strat])[:, pidx]
            dif_mgkgi = np.array(diff_permgkgibw_data[med][strat])[:, pidx]

            cand = canonical_doses[med]

            if strat in ['Age', 'Sex']:
                to_plot = dif_mgkgi * (cand / canonical_weight)
                yunit = f'(per {cand*unit_mult:0.0f} {unit} / {canonical_weight:0.0f} kg)'
            elif strat == 'Weight':
                to_plot = dif_mg * cand
                yunit = f'(per {cand*unit_mult:0.0f} {unit})'
            elif strat == 'Dose':
                to_plot = dif_abs
                yunit = f''
            
            # the dose readout is an exception, should always just be med units
            if pvar == 'dose':
                to_plot = dif_abs * unit_mult
                yunit = unit

            ax.plot(to_plot,
                    marker='o',
                    markersize=7,
                    color=colors.get(pvar, 'k'),
                    lw=0)

            ax.set_xticks(np.arange(len(to_plot))) 

            svals = cuts[strat][1] if strat!='Dose' else dose_cuts[med]
            mult = None if strat!='Dose' else canonical_weight * unit_mult
            xtl = parse_cutlabs(svals, strat, mult) if strat!='Dose' else dose_str_labs(len(svals))
            ax.set_xticklabels(xtl,
                               rotation=90)
            
            delt = '∆' if pname!='dose' else ''
            ax.set_ylabel(f'{delt}{pname}\n{yunit}', fontsize=FS)
            
            if pvi == len(pvars_todo)-1:
                ax.set_xlabel(f'{strat}\n{strat_units[sidx]}', fontsize=FS)
                ax.get_xaxis().set_label_coords(0.5,-0.5)

            ax.margins(0.1, 0.2)
            ax.tick_params(labelsize=FS)


    fig.savefig(f'figs/mags_{med.replace("meds_","")}.pdf')
    pl.close(fig)
        
           
# med co-occurrence analysis
pad = [5, 15]
fig, axs = pl.subplots(len(meds_use), len(meds_use), figsize=(15, 9),
                       gridspec_kw=dict(left=0.12,
                                        right=0.83,
                                        hspace=0.5,
                                        wspace=0.5,
                                        bottom=0.08,
                                        top=0.9),
                       sharex=True, sharey='row')
gas_fig, gas_axs = pl.subplots(len(gases), len(meds_use), figsize=(15, 6),
                               sharex=True, sharey='row')

for m0i, med_name in enumerate(meds_use):
        
    index_pts  = medication_to_bolus_index_points(data, med_name,
                                                       other_valid=valid,
                                                       return_doses=False,
                                                       offset=0)
    index_pts -= pad[0]

    _, horizon_data = parse_into_contexts_and_horizons(data,
                                                       index_pts,
                                                       max_horizon=pad[1]+pad[0],
                                                       context_length=1,
                                                       context_columns=meds_use,
                                                       horizon_columns=meds_use+gases,
                                                    )
    dat = horizon_data.mean(axis=0) # med x time

    for m1i, med1_name in enumerate(meds_use):
        ax = axs[m1i, m0i]
        
        ys = dat[m1i]
        xs = np.arange(len(ys)) - pad[0]
        ax.plot(xs, ys, color='k')

        ax.tick_params(labelsize=FS)

        if m1i == 0:
            mn0 = med_name.replace('meds_','').capitalize()
            ax.set_title(mn0, fontsize=FS, pad=15)
        if m0i == len(meds_use)-1:
            mn = med1_name.replace('meds_','').capitalize()
            ax.text(1.1, 0.5, mn, rotation=0, fontsize=FS, ha='left', va='center', transform=ax.transAxes)

    his = [0.7, 0.07, 0.4 , 2]
    for gi, gas_name in enumerate(gases):
        ax = gas_axs[gi, m0i]
        ix = gi + len(meds_use) # index in horizon data
        
        ys = dat[ix]
        xs = np.arange(len(ys)) - pad[0]
        ax.plot(xs, ys, color='k')

        ax.tick_params(labelsize=FS)

        if gi == 0:
            mn0 = med_name.replace('meds_','').capitalize()
            ax.set_title(mn0, fontsize=FS, pad=15)
        if m0i == len(meds_use)-1:
            gn = gas_name.replace('phys_','').replace('_exp_%','').capitalize()
            ax.text(1.1, 0.5, gn, rotation=0, fontsize=FS, ha='left', va='center', transform=ax.transAxes)

        ax.set_ylim([0, his[gi]])
        ax.set_yticks([0, his[gi]])
    
fig.text(0.045, 0.5, 'Medication #2\ndose (mg)', ha='center', va='center', rotation=0, fontsize=FS)
fig.text(0.5, 0.02, 'Minutes from medication #1 administration', ha='center', va='center', fontsize=FS)
fig.text(0.5, 0.98, 'Medication #1', ha='center', va='center', fontsize=FS, weight='bold')
fig.text(0.94, 0.5, 'Medication #2', ha='center', va='center', fontsize=FS, weight='bold')
fig.savefig(f'/Users/bdd/Desktop/cooc.pdf')
pl.close(fig)

gas_fig.text(0.045, 0.5, 'Gas %', ha='center', va='center', rotation=0, fontsize=FS)
gas_fig.text(0.5, 0.02, 'Minutes from medication administration', ha='center', va='center', fontsize=FS)
gas_fig.text(0.5, 0.98, 'Medication', ha='center', va='center', fontsize=FS, weight='bold')
gas_fig.text(0.98, 0.5, 'Gas', ha='center', va='center', fontsize=FS, weight='bold')

gas_fig.savefig(f'/Users/bdd/Desktop/cooc2.pdf')
pl.close(gas_fig)

# bolus timing analysis
fig, axs = pl.subplots(3,3,
                       figsize=(6,5),
                       sharex=True,
                       sharey=True,
                       gridspec_kw=dict(left=0.19,
                                        right=0.98,
                                        bottom=0.11,
                                        top=0.9,
                                        hspace=0.6,
                                        wspace=0.7
                                        ))
axs = axs.ravel()

all_valid = select_subset(data,
                      require_context=False,
                      require_horizon=False,
                      exclude_peri_airway=False,
                      return_context_and_horizon=False)

for med_name, ax in zip(meds_use, axs):

    index_pts = medication_to_bolus_index_points(data, med_name,
                                                       other_valid=all_valid,
                                                       return_doses=False,
                                                       offset=0)
    mins = data.minutes_idx.values[index_pts]
    allmins = data.minutes_idx.values

    p_by_min, edges = np.histogram(mins, bins=25, range=(0, 60*6))
    p_min, edges = np.histogram(allmins, bins=edges, range=(0, 60*6))
    p_by_min = p_by_min.astype(float) / p_min.astype(float)
    edges = (edges[1:]+edges[:-1])/2
    ax.bar(edges, p_by_min, color='k', width=np.diff(edges)[0])
    ax.set_yscale('log')
    ax.set_title(med_name.replace('meds_','').capitalize(), fontsize=FS)
    ax.margins(x=0.01)
    ax.tick_params(labelsize=FS)

fig.text(0.001, 0.5, 'p(bolus)', ha='left', fontsize=FS)
fig.text(0.5, 0.01, 'Mins in surgery', ha='center', fontsize=FS)
fig.savefig(f'figs/bolus_timing.pdf')
pl.close(fig)

