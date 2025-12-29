import pandas as pd
import numpy as np
import pickle
from matplotlib.gridspec import GridSpec
from or_foundational.params import input_features, output_features, keep_feat, colors, meds, meds_use, nicknames
from or_foundational.params.functions import load_data, select_subset, parse_into_contexts_and_horizons, medication_to_bolus_index_points, fold_ids_to_case_ids, compute_naive_predictions, generate_crossval_preds
FS = 10

data_path = 'data/model_input_data.h5'
fold_path = 'kfolds/model_kfold.npz'
model_path = 'models/med_model.pickle'

data = load_data(data_path)
valid = select_subset(data,
                      require_context=True,
                      require_horizon=True,
                      return_context_and_horizon=False)

all_contexts = {}
all_baselines = {}
all_horizons = {}
all_preds = {}
all_naives = {}
all_idx_pts = {}

for med_name in meds_use:

    index_pts = medication_to_bolus_index_points(data, med_name, other_valid=valid)
    context_data, horizon_data = parse_into_contexts_and_horizons(data,
                                                                  index_pts,
                                                                  context_columns=input_features,
                                                                  horizon_columns=output_features)
    context_data = context_data[:, keep_feat]
    baseline_data,_ = parse_into_contexts_and_horizons(data,
                                                      index_pts,
                                                      context_columns=output_features,
                                                      context_length=1,
                                                      horizon_columns=output_features)

    naives = compute_naive_predictions(data,
                                       index_pts,
                                       subtract_baseline=True)

    preds = generate_crossval_preds(model_path,
                                    fold_path,
                                    context_data,
                                    data.iloc[index_pts].case_id.values,
                                    )

    all_contexts[med_name] = context_data
    all_baselines[med_name] = baseline_data
    all_horizons[med_name] = horizon_data
    all_preds[med_name] = preds
    all_naives[med_name] = naives
    all_idx_pts[med_name] = index_pts

for output_feature in output_features:
    gs = GridSpec(3, 5,
                  wspace=0.6,
                  hspace=0.6,
                  width_ratios=[3,0.1, 1,1,1],
                  right=0.98,
                  )
    fig = pl.figure(figsize=(8,5))

    ofi = output_features.index(output_feature)
    col = colors[output_feature]

    gsy = [0,0,0,1,1,1,2,2,2]
    gsx = [2,3,4,2,3,4,2,3,4]
    yticks = [[0, 10],
              [-12, 0],
              [-12, 0],
              [0, 10],
              [-5, 0],
              [-5, 0],
              [-15, 0],
              [0, 10],
              [-15, 0]
              ]
    agg_naive_err = []
    agg_pred_err = []
    for med_idx, med_name in enumerate(meds_use):
        baseline_data = all_baselines[med_name].copy()
        horizon_data = all_horizons[med_name].copy()
        preds = all_preds[med_name].copy()
        naives = all_naives[med_name].copy()

        pred = preds[:, ofi, :]
        true_abs = horizon_data[:, ofi, :]
        naive = naives[:, ofi, :]

        pred -= baseline_data[:, ofi][...,None]
        true = true_abs - baseline_data[:, ofi][...,None]

        pred_mae = np.abs(pred - true)
        naive_mae = np.abs(naive - true)
        pred_err = pred_mae
        naive_err = naive_mae 
        agg_pred_err.append(pred_err)
        agg_naive_err.append(naive_err)

        ax = fig.add_subplot(gs[gsy[med_idx], gsx[med_idx]])
        ax.plot(pred.mean(axis=0), color='dimgrey', lw=1, label='Model')
        ax.plot(true.mean(axis=0), color=colors[output_feature], lw=1, zorder=1, label='Data')

        ax.tick_params(labelsize=FS)
        if gsy[med_idx] != gs.nrows - 1:
            ax.set_xticklabels([])

        ax.set_title(med_name.replace('meds_', '').capitalize(), fontsize=FS)

        if med_idx == 3:
            ax.set_ylabel(f'∆ {nicknames[output_feature]}\n(mmHg)', fontsize=FS,
                          ha='center',
                          va='center',
                          rotation=0,
                          labelpad=25)

        if med_idx == 1:
            ax.legend(fontsize=FS-1, frameon=False, loc=(0.4, 0.4),
                      labelcolor='linecolor',
                      handlelength=0)

    ax = fig.add_subplot(gs[:, 0:1])
    err_pred = np.mean(np.concatenate(agg_pred_err, axis=0), axis=0)
    err_naive = np.mean(np.concatenate(agg_naive_err, axis=0), axis=0)
    ax.plot(err_naive, color=[.74,.77,.77], label='Naive', ls='--')
    ax.plot(err_pred, color='dimgrey', label='Model')

    ax.set_ylabel('error', fontsize=FS, rotation=0, labelpad=25)
    ax.tick_params(labelsize=FS)
    ax.legend(fontsize=FS, frameon=False)

    fig.text(0.5, 0.03, 'Minutes from medication',
             fontsize=FS,
             ha='center',
             va='center')

    for ax in fig.axes:
        ax.set_xticks([0, 15])
        ax.tick_params(length=5, width=0.5)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(0.5)
        
    fig.savefig(f'figs/model_perf_{nicknames[output_feature]}.pdf', dpi=200)
    pl.close(fig)

for med_idx, med_name in enumerate(meds_use):

    context_data = all_contexts[med_name]
    baseline_data = all_baselines[med_name]
    horizon_data = all_horizons[med_name]
    preds = all_preds[med_name]
    naives = all_naives[med_name]
    
    fig, axs = pl.subplots(4, len(output_features),
                           figsize=(15,9),
                           gridspec_kw=dict(hspace=0.4))

    for ofi, output_feature in enumerate(output_features):
        pred = preds[:, ofi, :]
        trues = horizon_data[:, ofi, :]
        naive = naives[:, ofi, :]

        pred -= baseline_data[:, ofi][...,None]
        trues -= baseline_data[:, ofi][...,None]

        col = colors[output_feature]

        ax = axs[0, ofi]
        ax.plot(pred.mean(axis=0), color=col)
        ax.plot(trues.mean(axis=0), color='grey', lw=2, alpha=0.8)

        ax = axs[1, ofi]
        mae = np.abs(pred - trues).mean(axis=0)
        mae_naive = np.abs(naive - trues).mean(axis=0)
        ax.plot(mae, color=col)
        ax.plot(mae_naive, color=col, ls=':')
        
        ax = axs[2, ofi]
        nmins = 5
        tvals = trues[:, :nmins].ravel()
        pvals = pred[:, :nmins].ravel()
        ax.scatter(tvals, pvals, color=col, s=1)
        r = np.corrcoef(tvals, pvals)[0,1]
        ax.set_title(f'r={r:0.2f}', fontsize=10)

        model_err = np.abs(pred - trues).ravel()
        naive_err = np.abs(naive - trues).ravel()
        ax = axs[3, ofi]
        ax.scatter(model_err, naive_err, color=col, s=1)
        ax.set_title(f'{100*np.mean(model_err<naive_err):0.0f}% model beats naive', fontsize=10)

    fig.savefig(f'figs/model_{med_name}.jpg', dpi=200)
    pl.close(fig)

