##
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
from pandas.api.types import is_numeric_dtype

from or_foundational.params.params import clip_limits

def process_var(name, vals):
    vals[np.isnan(vals)] = np.nanmean(vals)

    if name in clip_limits:
        lim = clip_limits[name]
        if isinstance(lim, list):
            minn, maxx = lim
        else:
            minn = np.min(vals)
            maxx = lim
        vals[vals<minn] = minn
        vals[vals>maxx] = maxx
    return vals

##
def curate(case):

    case.replace([np.inf, -np.inf], np.nan, inplace=True)
    for idx, (column, dtype) in enumerate(case.dtypes.items()):
        if not is_numeric_dtype(dtype):
            continue
        case[column] = process_var(column, case[column].values)
    if len(case) > 20 * 60:
        case = case.iloc[:20*60]
    return case
##
