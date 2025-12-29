# conda activate gpu
import pandas as pd
import numpy as np
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from cuml.linear_model import LinearRegression
import xgboost as xgb

from or_foundational.params import context, max_horizon, meds, gases, phys, demog, input_features, output_features, keep_feat
from or_foundational.params.functions import load_data, select_subset, parse_into_contexts_and_horizons, medication_to_bolus_index_points, fold_ids_to_case_ids


outname = 'med_model'
train_chunk_size = 12000000
data_path = 'data/model_input_data.h5'
kfold_path = 'kfolds/model_kfold.npz'

all_models = {}
for holdout in range(3):
    
    fold_ids = [i for i in range(3) if i!=holdout]
    case_ids = fold_ids_to_case_ids(kfold_path, fold_ids)
    
    print(f'Loading fold {holdout}.')
    data = load_data(data_path, case_ids=case_ids)
    print(f'Subsetting and parsing.')
    valid = select_subset(data,
                          require_context=True,
                          require_horizon=True,
                          return_context_and_horizon=False)
    index_pts = np.where(valid)[0]
    context_data, horizon_data = parse_into_contexts_and_horizons(data,
                                                                  index_pts,
                                                                  context_columns=input_features,
                                                                  horizon_columns=output_features)
    context_data = context_data[:, keep_feat]
   
    holdout_models = {}
    for ofi, output_feature in enumerate(output_features):

        xgb_models, lr_models = [], []
        for horizon in range(max_horizon):
            print(f'holdout={holdout} feat={output_feature} horizon={horizon}')

            x = context_data
            y = horizon_data[:, ofi, horizon]

            xgb_model = xgb.XGBRegressor(device='cuda')
            booster = None
            for train_batch_idx in range(0, len(x), train_chunk_size):
                print(f'\tTraining chunk {train_batch_idx} - {train_batch_idx + train_chunk_size} / {len(x)}')
                xgb_model.fit(x[train_batch_idx:train_batch_idx+train_chunk_size],
                              y[train_batch_idx:train_batch_idx+train_chunk_size],
                              xgb_model=booster)
                booster = xgb_model.get_booster()
            xgb_models.append(xgb_model)
            
            #lr_model = LinearRegression(fit_intercept=True, copy_X=True)
            #lr_model.fit(x, y)
            #model_rep = np.append(lr_model.coef_, lr_model.intercept_)
            #lr_models.append(model_rep)
            lr_models.append(None)

        holdout_models[output_feature] = dict(xgb=xgb_models, lr=lr_models)
    all_models[holdout] = holdout_models

with open(f'models/{outname}.pickle', 'wb') as f:
    pickle.dump(all_models, f, protocol=pickle.HIGHEST_PROTOCOL)

##
