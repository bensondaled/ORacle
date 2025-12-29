import pandas as pd
import numpy as np
import os
pj = os.path.join

data_path = 'data/query/'
sheet = 'aims_intraopnotes.csv'
note_data = pd.read_csv(pj(data_path, sheet))
keep_ids = [50117, #intubation  
            50127, # extubation awake
            50202, # extubation
            50145, # LMA removed
            'lma inserted',
        ]

criteria = []
for kid in keep_ids:
    if isinstance(kid, (int, float)):
        keep = note_data.mpog_note_concept_id == kid
    else:
        keep = note_data.aims_note_concept_desc.str.lower().str.strip().str.contains(kid)
    criteria.append(keep)
criteria = np.logical_or.reduce([i.values for i in criteria])
note_data = note_data.iloc[criteria]

note_data.to_hdf(pj(data_path, 'processed_notes.h5'), key='notes', format='table', data_columns=['mpog_case_id'])

def process_note(grp):

    desc = grp.aims_note_concept_desc
    assert len(desc.dropna().unique()) == 1
    desc = desc.iloc[0]

    ts = grp.aims_user_entered_ts
    assert len(ts.dropna().unique()) < 2, print(ts)
    ts = ts.iloc[0]

    comps = []
    for _,row in grp.iterrows():
        comp_desc = row.aims_note_concept_desc
        comp_val0 = row.aims_value_text
        comp_val1 = row.aims_value_numeric
        comp_val2 = row.aims_value_cd
        comp_val = '/'.join([v for v in [comp_val0, comp_val1, comp_val2] if isinstance(v, str)])
        comps.append(f'{comp_desc}: {comp_val}')
    
    comps = '--'.join(comps)
    note_txt = f'{desc}:: {comps}'
    note_kind = desc

    case_id = grp.mpog_case_id
    assert len(case_id.dropna().unique()) < 2
    case_id = case_id.iloc[0]
    mpog_case_id = case_id

    return pd.Series(dict(note_ts=ts, note_kind=note_kind, note_txt=note_txt, mpog_case_id=mpog_case_id))

processed = note_data.groupby('mpog_note_id').apply(process_note, include_groups=False)
processed.to_csv(pj(data_path, 'processed_notes.csv'))

with pd.HDFStore(pj(data_path, 'processed_notes.h5')) as h:
    notes = h.notes
intub = notes[notes.note_kind.isin(['Simple Airway', 'Complex Airway', 'Intubation'])]
intub_ids = intub.mpog_case_id.values
np.save('data/intubation_cases', intub_ids)
