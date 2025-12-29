##
# Launch postgres app before running this

import numpy as np
import pandas as pd
import pickle, gzip, json, os
import psycopg2 as psycopg
import subprocess as sp
from datetime import datetime
import string
import matplotlib.colors as mcolors

from or_foundational.params.functions import generate_crossval_preds
from or_foundational.params.params import keep_feat, keep_feat_names, input_features
from clinician_years import clinician_years

def safe_str(s):
    for char in string.punctuation:
        s = s.replace(char, '_')
    return s.lower()

def download_latest_data():

    analysis_port = '5431' # entered in the postgres gui (made an analysis-specific one to keep separate from testing the website locally)
    DB_CONNECT_STR = f'dbname=bdd user=bdd port={analysis_port}' # 5431 for clinician-testing-analysis
    DB_CONNECT_KW = dict(dsn=DB_CONNECT_STR)

    ''' only needed to do once ever
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS clinician_responses (
        id SERIAL PRIMARY KEY,
        user_id TEXT NOT NULL,
        question_key TEXT NOT NULL,
        response_values JSONB,
        text TEXT NOT NULL,
        timestamp TEXT NOT NULL
    );
    """
    with psycopg.connect(**DB_CONNECT_KW) as conn:
        with conn.cursor() as cur:
            cur.execute(create_table_query)
            conn.commit()
    '''

    data_storage_path = '/Users/bdd/data/or-foundational/clinician_testing'

    dtstr = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    dumpname = f'downloaded_{dtstr}'
    dumpfile = f'{dumpname}.dump'
    grab_data_cmds = [
            'heroku open --app clinician-testing'
            'heroku pg:backups:capture --app clinician-testing', # somtimes you have to just copy and paste this into a terminal and run separately, no idea why. (and then can just run this all again)
            'heroku pg:backups:download --app clinician-testing',
            f'mv latest.dump {dumpfile}',
            f'pg_restore --verbose --clean --no-acl --no-owner -h localhost -p {analysis_port} -U bdd -d bdd {dumpfile}',
            ]
    for cmd in grab_data_cmds:
        result = sp.run(cmd, capture_output=True, text=True, shell=True, cwd=data_storage_path)
        print(f"Command: {' '.join(cmd)}")
        print(f"Output:{result.returncode}\n{result.stdout}\n{result.stderr}")

    with psycopg.connect(**DB_CONNECT_KW) as conn:
        query = f'SELECT * FROM clinician_responses'
        rows = pd.read_sql_query(query, conn)
    
    csv_path = os.path.join(data_storage_path, f'{dumpname}.csv')
    rows.to_csv(csv_path, index=False)
    return csv_path

## grab responses, and make model predictions

# download new data:
# run manually first: heroku pg:backups:capture --app clinician-testing
#csv_path = download_latest_data()
# or use an old download:
csv_path = '/Users/bdd/data/or-foundational/clinician_testing/downloaded_2025_09_02_11_11_35.csv'
responses = pd.read_csv(csv_path)

data_path = '/Users/bdd/code/or_foundational/clinician_testing/interface/data'
data_path_cases_numerical = os.path.join(data_path, 'clinican_test_data_examples.pickle')
test_variable = 'phys_bp_mean_non_invasive'

with gzip.open(data_path_cases_numerical, 'rb') as f1:
    data_cases = pickle.load(f1)

data_cases_context = {k.lower():v for k,v in data_cases['context'].items()}
data_cases_horizon = data_cases['horizon']
data_cases_horizon = {str(k).lower():v for k,v in data_cases_horizon.items()}

'''
emails = [
          'lhan2@stanford.edu',
          'chungp@stanford.edu',
          'agoodell@stanford.edu',
          #'naghaeep@stanford.edu',
          'gbrice@stanford.edu',
          'ergross@stanford.edu',
          ]
'''
#emails_exclude = ['ben', 'marcgh@stanford.edu', 'marcghh@stanford.edu']
with open('/Users/bdd/code/or_foundational/clinician_testing/interface/data/clinician_id_key.txt','r') as f:
    d = eval(f.read())
    valid_emails = list(d.keys())
    valid_emails += ['teresan1@stanford.edu']
emails_exclude = [e for e in responses.user_id.unique() if e not in valid_emails]
ex = responses[~responses.user_id.isin(emails_exclude)]
ex_all = ex.groupby('user_id').apply(lambda x: x.drop_duplicates(subset='question_key', keep='last')).reset_index(drop=True)
resamp=3

##
'''
def precompute_model_preds(qk):
    stripped = qk.replace('images/case_figs/','').replace('.jpg','')

    cid = 0 # will apply only the mean contexts
    mod_con_dat = data_cases_context[stripped]

    if '_' not in stripped: # was a single individual case, need to trim according to min_idx
        min_idx = int(stripped)
        mod_con_dat = mod_con_dat.loc[:min_idx+1]
        mod_con_dat = mod_con_dat.iloc[-5:]
        cid = data_cases_context[stripped].case_id.unique()[0]

    # to be extra fair
    mod_con_dat = mod_con_dat.reset_index(drop=True)
    mod_con_dat.loc[0, test_variable] = mod_con_dat.loc[1, test_variable]
    mod_con_dat.loc[2, test_variable] = mod_con_dat.loc[4, test_variable]
    mod_con_dat.loc[3, test_variable] = mod_con_dat.loc[4, test_variable]

    mod_con_dat = mod_con_dat[input_features].values.T.ravel() # first transpose to make it features x timepoints, then it will align with the keep_feat ordering
    mod_con_dat = mod_con_dat[keep_feat]
    mod_con_dat = mod_con_dat[None, :]
    mod_pred = generate_crossval_preds(model_path='/Users/bdd/data/or-foundational/med_modeling/med_model_5-31.pickle',
                                       fold_path='/Users/bdd/data/or-foundational/med_modeling/kfold_2025-08-14_16-39-14.npz',
                                       context_data=mod_con_dat,
                                       cids=np.array([cid]), 
                                       pred_features=[test_variable]
                                       )
    mod_pred = mod_pred[0,0,:] # all samples (ie 1 only) and all output features (just testing one)
    mod_pred = mod_pred[2::resamp] # 2 because index 0 is the min after the bolus
    return mod_pred
uqk = ex_all.question_key.unique()
mod_preds = {qk:precompute_model_preds(qk) for qk in uqk}
with open('/Users/bdd/data/or-foundational/clinician_testing/mod_preds_cache_2025-11-15.pickle','wb') as f:
    pickle.dump(mod_preds, f)
'''
with open('/Users/bdd/data/or-foundational/clinician_testing/mod_preds_cache_2025-11-15.pickle','rb') as f:
    mod_preds = pickle.load(f)

## model vs true
fig, ax = pl.subplots()
all_mae = []
all_trues = {}
for qk in mod_preds.keys():

    # NOTE : use only the mean, or only indiv examples
    #is_mean_style = qk.startswith('images/case_figs/meds')
    #if is_mean_style:
    #    continue

    stripped = qk.replace('images/case_figs/','').replace('.jpg','')
    first_true_val = data_cases_horizon[stripped][0]
    true = np.array(data_cases_horizon[stripped][3::resamp]) # 3 because index 0 is the value at time of delivery, and so minute 3 is then 3 ahead of that
    mod_pred = mod_preds[qk]
    
    mae = np.abs(mod_pred - true)
    all_mae.append(mae)
    ax.plot(mae, color='grey', lw=0.5)

    all_trues[qk] = np.append(first_true_val,true)

mean_mae = np.mean(all_mae, axis=0)
ax.plot(mean_mae, color='k', lw=2)

## quick verifications
qs = ex_all[~ex_all.question_key.str.contains('__')]
print(qs.question_key.unique().shape)
print(qs.shape)
print(ex_all.user_id.unique().shape)

## plot results - inspections
plot_indiv = True
agg = []

for uid in ex_all.user_id.unique():

    #if uid != 'lhan2@stanford.edu':
    #if uid != 'chungp@stanford.edu':
    #if uid != 'bjwbjw@stanford.edu':
    #    continue
    
    #if plot_indiv:
    #    fig, axs = pl.subplots(5, 6, sharex=True, sharey=True, num=uid)
    #    axs = axs.ravel()

    ex = ex_all[ex_all.user_id == uid]
    
    # NOTE : use only the mean, or only indiv examples
    is_mean_style = ex.question_key.str.startswith('images/case_figs/meds')
    ex = ex[~is_mean_style]
    #print(ex.question_key.unique())

    for idx, (_, info) in enumerate(ex.iterrows()):
        qk = info.question_key
        guess = json.loads(info.response_values)
        trueval = guess[0]
        guess = guess[1:] # cut out the first true value
        r = 3 if resamp == 1 else 1
        guess = np.repeat(guess, r)
        
        stripped = qk.replace('images/case_figs/','').replace('.jpg','')
        true = np.array(data_cases_horizon[stripped][3::resamp]) # 3 because index 0 is the value at time of delivery, and so minute 3 is then 3 ahead of that
        naive = np.ones_like(guess) * trueval
        mod_pred = mod_preds[qk]
        
        '''
        if plot_indiv:
            ax = axs[idx]
            #ax.plot(true, color='forestgreen', label='true')
            #ax.plot(guess, color='maroon', label='guess')
            #ax.plot(mod_pred, color='steelblue', label='model')

            mod_mae = np.abs(mod_pred - true)
            usr_mae = np.abs(guess - true)
            ax.plot(mod_mae, color='red', label='model mae')
            ax.plot(usr_mae, color='blue', label='user mae')

            #ax.plot(naive, color='grey', label='naive')
            ax.set_title(f'{stripped}', fontsize=8)
        '''

        #tx = ax.twinx()
        #tx.plot(np.abs(true-guess), color='red', ls=':')
        #tx.plot(np.abs(true-mod_pred), color='blue', ls=':')

        agg.append([true, guess, mod_pred, naive, uid])

    #if plot_indiv:
    #    ax.legend()

uids = np.array([a[-1] for a in agg])
t,g,m,n = np.array([a[:-1] for a in agg]).transpose([1,0,2])
agg_fxn = np.mean

if plot_indiv:
    fig, ax = pl.subplots()
    cols = list(mcolors.CSS4_COLORS.keys())
    np.random.shuffle(cols)
    for idx,u in enumerate(np.unique(uids)):
        keep = uids == u
        t_ = t[keep]
        g_ = g[keep]
        m_ = m[keep]
        n_ = n[keep]
        g_err = agg_fxn(np.abs(t_-g_), axis=0)
        m_err = agg_fxn(np.abs(t_-m_), axis=0)
        n_err = agg_fxn(np.abs(t_-n_), axis=0)
        g_sd = np.abs(t_-g_).std(axis=0) / np.sqrt(len(t_))
        m_sd = np.abs(t_-m_).std(axis=0) / np.sqrt(len(m_))
        n_sd = np.abs(t_-n_).std(axis=0) / np.sqrt(len(n_))

        x = np.arange(len(g_err)) * 3 + 3
        
        # show model
        lab = 'model' if idx == 0 else None
        ax.plot(x, m_err, color='k', label=lab, lw=2)
        # show naive
        lab = 'naive' if idx == 0 else None
        ax.plot(x, n_err, color='grey', label=lab, lw=2)

        # show human
        ax.plot(x, g_err, color=cols[idx], label=f'{u[:5]}')
        ax.fill_between(x, g_err-g_sd, g_err+g_sd, color=cols[idx], alpha=0.2, lw=0,)
    ax.set_ylabel('mae')
    ax.set_xlabel('minutes post med')
    ax.legend()
    ax.set_xticks(np.arange(3,16,3))

# aggregated meta fig
fig, ax = pl.subplots()
g_err = agg_fxn(np.abs(t-g), axis=0)
m_err = agg_fxn(np.abs(t-m), axis=0)
n_err = agg_fxn(np.abs(t-n), axis=0)
g_sd = np.abs(t-g).std(axis=0) / np.sqrt(len(t))
m_sd = np.abs(t-m).std(axis=0) / np.sqrt(len(m))
n_sd = np.abs(t-n).std(axis=0) / np.sqrt(len(n))

x = np.arange(len(g_err)) * 3 + 3

# show model
ax.plot(x, m_err, color='k', label='Model', lw=3)
ax.fill_between(x, m_err-m_sd, m_err+m_sd, color='grey', alpha=0.2, lw=0,)
# show naive
#ax.plot(x, n_err, color='grey', label='Naive', lw=3)

# show human
ax.plot(x, g_err, color='maroon', label='Clinicians', lw=3)
ax.fill_between(x, g_err-g_sd, g_err+g_sd, color='maroon', alpha=0.2, lw=0,)
ax.set_ylabel('MAE')
ax.set_xlabel('Minutes post-medication')
ax.legend()
ax.set_xticks(np.arange(3,16,3))

## subj-subj variance
fig, axs = pl.subplots(14, 14,
                       figsize=(15,9),
                       gridspec_kw=dict(left=0.03, right=0.99, bottom=0.04, top=0.99, wspace=0.5, hspace=0.5), sharex=True)
axs = axs.ravel()
idx = 0

quse = ex_all.question_key.unique()
quse = [q for q in quse if 'meds_' not in q]

xvals = [0,3,6,9,12,15]
for q in quse:
    subset = ex_all[ex_all.question_key == q]
    true = np.round(all_trues[q])
    m = mod_preds[q]
    if True:#len(subset) > 1:
        ax = axs[idx]
        all_resp = []
        for resp in subset.response_values.values:
            resp = eval(resp)
            ax.plot(xvals, resp, lw=0.5, color='grey')
            assert resp[0] == true[0]
            all_resp.append(resp)
        ax.plot(xvals, true, color='black', lw=1, label='true')
        ax.plot(xvals, np.append(resp[0], m), color='steelblue', lw=1, label='mod')
        mean_resp = np.mean(all_resp, axis=0)
        #ax.plot(mean_resp, color='dimgrey', lw=1.5, label='clin')
        #ax.text(0.4, 0.9, os.path.split(q)[-1], fontsize=6, transform=ax.transAxes)
        ax.tick_params(labelsize=6, width=0.5)
        idx += 1
#ax.legend()
axs[-10].set_xticks(xvals)
axs[-10].set_xticklabels(['0','','','','', '15'])
rm = len(axs.ravel()) - len(quse)
for ax in axs[-rm:]:
    ax.axis('off')
fig.text(0.005, 0.5, 'MAP (mmHg)', rotation=90, fontsize=8,)
fig.text(0.5, 0.005, 'Minutes', fontsize=8,)

## -- FORMAL FIGURES -- 

## First format into a clean table of results

def parse_guess(row):
    qk = row.question_key
    guess = json.loads(row.response_values)
    guess = guess[1:] # cut out the first true value
    r = 3 if resamp == 1 else 1
    guess = np.repeat(guess, r)
    return guess
def parse_true(row):
    qk = row.question_key
    stripped = qk.replace('images/case_figs/','').replace('.jpg','')
    true = np.array(data_cases_horizon[stripped][3::resamp])
    return true
def parse_mod(row):
    qk = row.question_key
    mod_pred = mod_preds[qk]
    return mod_pred
def parse_val0(row):
    qk = row.question_key
    stripped = qk.replace('images/case_figs/','').replace('.jpg','')
    v0 = np.array(data_cases_horizon[stripped][0])
    return v0

res = ex_all.copy()
res['q_type'] = res.question_key.str.startswith('images/case_figs/meds') # True if mean_style
res['guess'] = res.apply(parse_guess, axis=1)
res['true'] = res.apply(parse_true, axis=1)
res['modpred'] = res.apply(parse_mod, axis=1)
res['val0'] = res.apply(parse_val0, axis=1)

##

def A(x):
    return np.array([i for i in x])

def me(x, axis=0):
    if axis is None:
        x = x.ravel()
        axis = 0
    return np.nanmean(x, axis=axis), np.nanstd(x, ddof=1, axis=axis)/np.sqrt(x.shape[axis])

def summarize(qgrp):
    gues = A(qgrp.guess.values)
    tru = A(qgrp.true.values)
    mp = A(qgrp.modpred.values)
    v0 = A(qgrp.val0.values)
    
    assert np.all([x==v0[0] for x in v0])
    assert np.all(np.allclose(tru[0], tru))
    assert np.all(np.allclose(mp[0], mp))
    tru = tru[0]
    mp = mp[0]
    v0 = v0[0]

    mean_guess, sem_guess = me(gues, axis=0)
    
    guess_mae = np.mean(np.abs(gues - tru), axis=0)
    model_mae = np.abs(mp - tru)
    
    guess_max_delta = (gues-v0)[np.arange(len(gues)), np.argmax(np.abs(gues - v0), axis=1)]
    mod_max_delta = (mp-v0)[np.argmax(np.abs(mp - v0))]
    true_max_delta = (tru-v0)[np.argmax(np.abs(tru - v0))]

    return dict(mean_guess=mean_guess,
                true=tru,
                model=mp,
                guess_mae=guess_mae,
                model_mae=model_mae,
                guess_max_delta=guess_max_delta,
                mod_max_delta=mod_max_delta,
                true_max_delta=true_max_delta)


## new version
fig, axs = pl.subplots(1, 3, gridspec_kw=dict(wspace=0.8, left=0.14),
                       figsize=(14,5))

r0 = res[res.q_type == 0] # indiv examples only

summary = r0.groupby('question_key').apply(summarize, include_groups=False).apply(pd.Series).reset_index()
g_mae = A(summary.guess_mae.values)
m_mae = A(summary.model_mae.values)
tru = A(summary.true.values)
pct_g_mae = 100* (g_mae / tru)
pct_m_mae = 100* (m_mae / tru)

# MAE trajectory
g_mae_mean, g_mae_sem = me(pct_g_mae, axis=0)
m_mae_mean, m_mae_sem = me(pct_m_mae, axis=0)
ax = axs[0]
ax.fill_between([3,6,9,12,15],
                g_mae_mean-g_mae_sem,
                g_mae_mean+g_mae_sem,
                color='k', lw=0, alpha=0.3)
ax.fill_between([3,6,9,12,15],
                m_mae_mean-m_mae_sem,
                m_mae_mean+m_mae_sem,
                color='steelblue', lw=0, alpha=0.3)
ax.plot([3,6,9,12,15], g_mae_mean, label='Clinicians', color='k', lw=3)
ax.plot([3,6,9,12,15], m_mae_mean, label='Model', color='steelblue', lw=3)
ax.legend()
ax.set_ylabel('MAE (%)')

# abs error bars
g_mean, g_err = me(pct_g_mae, axis=None)
m_mean, m_err = me(pct_m_mae, axis=None)

# for stat values
_g = pct_g_mae.mean(axis=1) # 1 per-question for proper stats
_m = pct_m_mae.mean(axis=1)
from scipy.stats import mannwhitneyu
print(_g.mean(), _g.std())
print(_m.mean(), _m.std())
print(mannwhitneyu(_g, _m).pvalue)

ax = axs[1]
ax.bar([0,], [g_mean], yerr=[g_err], color='k', label='Clinician', ecolor='grey')
ax.bar([1,], [m_mean], yerr=[m_err], color='steelblue', label='Model', ecolor='grey')
ax.set_ylabel('MAE (%)')
ax.set_xticks([0,1])
ax.set_xticklabels(['Clinicians', 'Model'])

# max deltas
tmd = A(summary.true_max_delta.values)
gmd = np.array([np.mean(s) for s in summary.guess_max_delta.values])
mmd = A(summary.mod_max_delta.values)

neg_change = tmd<0
pos_change = tmd>=0

g_md_, g_md_e = me(np.abs(tmd - gmd))
m_md_, m_md_e = me(np.abs(tmd - mmd))
g_md_n_, g_md_n_e = me(np.abs(tmd[neg_change] - gmd[neg_change]))
m_md_n_, m_md_n_e = me(np.abs(tmd[neg_change] - mmd[neg_change]))
g_md_p_, g_md_p_e = me(np.abs(tmd[pos_change] - gmd[pos_change]))
m_md_p_, m_md_p_e = me(np.abs(tmd[pos_change] - mmd[pos_change]))

ax = axs[2]
ax.bar([0,3,6], [g_md_, g_md_n_, g_md_p_], yerr=[g_md_e, g_md_n_e, g_md_p_e],
       color='k', label='Clinician', ecolor='grey')
ax.bar([1, 4, 7], [m_md_, m_md_n_, m_md_p_], yerr=[m_md_e, m_md_n_e, m_md_p_e],
       color='steelblue', label='Model', ecolor='grey')
ax.set_xticks([0.5, 3.5, 6.5])
ax.set_xticklabels(['All', 'Neg ∆', 'Pos ∆'], )
ax.set_ylabel('MAE, peak effect estimate')

axs[0].set_xticks([3,6,9,12,15])
axs[0].set_xlabel('Minutes from bolus')



## scatter
def ccc(x,y):
    cor = np.corrcoef(x, y)[0][1]
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    sd_x = np.std(x, ddof=1)
    sd_y = np.std(y, ddof=1)
    numerator = 2 * cor * sd_x * sd_y
    denominator = var_x + var_y + (mean_x - mean_y)**2
    ccc = numerator / denominator
    return ccc

r0 = res[res.q_type == 0] # indiv examples only
mod = np.array([i for i in r0.modpred.values])
gues = np.array([i for i in r0.guess.values])
tru = np.array([i for i in r0.true.values])

fig,axs = pl.subplots(1,3, sharex=True, sharey=True, figsize=(9,2.5),
                      gridspec_kw=dict(wspace=0.75, bottom=0.25))

ds = {'Data':tru, 'Clinicians':gues, 'Model':mod}
for idx, (a,b) in enumerate([['Data', 'Model'], ['Data', 'Clinicians'], ['Clinicians', 'Model']]):
    ax = axs[idx]
    #r = (np.corrcoef(ds[a].ravel(), ds[b].ravel())[0,1]) ** 1
    r = (ccc(ds[a].ravel(), ds[b].ravel()))
    ax.scatter(ds[a].ravel(), ds[b].ravel(), color='k', s=3, label=f'r={r:0.2f}')
    ax.legend(fontsize=10, frameon=False, loc='lower right')
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.set_xlim([40,150])
    ax.set_ylim([40,150])
    ax.set_xticks([50,100,150])
    ax.set_yticks([50,100,150])
    ax.plot([40,160],[40,150],color='grey',lw=0.25)

## experience

r0 = res.copy()
r0['grad'] = r0.user_id.map(clinician_years)
r0['yrs'] = 2025 - r0.grad
r0['guess_mae'] = np.abs(r0.guess - r0.true)
pts = []
for subj in r0.user_id.unique():
    s = r0[r0.user_id == subj]
    mae = s.guess_mae
    mae = np.array([_ for _ in mae.values])
    mean = np.mean(mae)
    sd = np.std(mae)
    yrs = s.yrs.iloc[0]
    pts.append([yrs, mean, sd])
pts = np.array(pts).T

fig, ax = pl.subplots()
ax.errorbar(pts[0], pts[1], yerr=pts[2], lw=0,
            #elinewidth=1,
            marker='o', markersize=5)

##--- Old version

fig, all_axs = pl.subplots(2, 3, gridspec_kw=dict(wspace=0.8, left=0.14),
                       figsize=(14,5))
qstr = {0:'Individual\nscenarios', 1:'Mean\nscenarios'}
for q_type in [0,1]:
    axs = all_axs[q_type]

    r0 = res[res.q_type == q_type]
    summary = r0.groupby('question_key').apply(summarize, include_groups=False).apply(pd.Series).reset_index()
    g_mae = A(summary.guess_mae.values)
    m_mae = A(summary.model_mae.values)
    tru = A(summary.true.values)
    pct_g_mae = 100* (g_mae / tru)
    pct_m_mae = 100* (m_mae / tru)

    g_mean, g_err = me(pct_g_mae, axis=None)
    m_mean, m_err = me(pct_m_mae, axis=None)

    # MAE trajectory
    g_mae_mean, g_mae_sem = me(g_mae, axis=0)
    ax = axs[0]
    ax.fill_between([3,6,9,12,15],
                    g_mae_mean-g_err,
                    g_mae_mean+g_err,
                    label='Clinicians', color='k', lw=0, alpha=0.3)
    ax.plot([3,6,9,12,15], g_mae.mean(axis=0), label='Clinicians', color='k', lw=3)
    ax.plot([3,6,9,12,15], m_mae.mean(axis=0), label='Model', color='grey', lw=3)
    if q_type==0:
        ax.text(5.1, 14.5, 'Clinicians')
        ax.text(5.1, 10.5, 'Model', color='grey')
    
    # abs error bars
    ax = axs[1]
    ax.bar([0,], [g_mean], yerr=[g_err], color='k', label='Clinician', ecolor='lightgrey')
    ax.bar([1,], [m_mean], yerr=[m_err], color='grey', label='Model', ecolor='lightgrey')
    ax.set_ylabel('% absolute error\n(all pts)')
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Clinicians', 'Model'])

    # max deltas
    tmd = A(summary.true_max_delta.values)
    gmd = np.array([np.mean(s) for s in summary.guess_max_delta.values])
    mmd = A(summary.mod_max_delta.values)

    neg_change = tmd<0
    pos_change = tmd>=0

    g_md_, g_md_e = me(np.abs(tmd - gmd))
    m_md_, m_md_e = me(np.abs(tmd - mmd))
    g_md_n_, g_md_n_e = me(np.abs(tmd[neg_change] - gmd[neg_change]))
    m_md_n_, m_md_n_e = me(np.abs(tmd[neg_change] - mmd[neg_change]))
    g_md_p_, g_md_p_e = me(np.abs(tmd[pos_change] - gmd[pos_change]))
    m_md_p_, m_md_p_e = me(np.abs(tmd[pos_change] - mmd[pos_change]))

    ax = axs[2]
    ax.bar([0,3,6], [g_md_, g_md_n_, g_md_p_], yerr=[g_md_e, g_md_n_e, g_md_p_e],
           color='k', label='Clinician', ecolor='lightgrey')
    ax.bar([1, 4, 7], [m_md_, m_md_n_, m_md_p_], yerr=[m_md_e, m_md_n_e, m_md_p_e],
           color='grey', label='Model', ecolor='lightgrey')
    ax.set_xticks([0.5, 3.5, 6.5])
    ax.set_xticklabels(['All', 'Neg ∆', 'Pos ∆'], )
    ax.set_ylabel('Abs err\n(peak effect)')
    if q_type == 0:
        ax.legend(fontsize=6)
    
    axs[0].text(-0.4, 0.5, f'{qstr[q_type]}', ha='center', va='center',
                transform=axs[0].transAxes)

all_axs[1,0].set_xlabel('Minutes from bolus')
##
