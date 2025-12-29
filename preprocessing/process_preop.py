import pandas as pd
import os
pj = os.path.join
import numpy as np

data_path = 'data/query/'

concepts = {
        'weight' : 70264, # weight in kg
        'weight_lb' : 70265, # weight in lbs
        'height' : 70257, # height in cm
        'height_in' : 70258, # height in in
        }


result = {}

for chunk_idx,chunk in enumerate(pd.read_csv(pj(data_path, 'aims_preop.csv'), chunksize=100000)):
    print(chunk_idx)
    for concept_name, idd in concepts.items():
        x = chunk[chunk.mpog_preop_concept_id == idd]

        if idd == 70258:
            x['aims_value_numeric'] = x.aims_value_numeric.values * 2.54
            concept_name = 'height'
        elif idd == 70265:
            x['aims_value_numeric'] = x.aims_value_numeric.values / 2.2
            concept_name = 'weight'

        if len(x) != 0:
            for _,row in x.iterrows():
                case_id = row.mpog_case_id
                if case_id not in result:
                    result[case_id] = {}
                result[case_id][concept_name] = row.aims_value_numeric

result = [[k,v] for k,v in result.items()]
[r[1].update(mpog_case_id=r[0]) for r in result]
result = [r[1] for r in result]
result = pd.DataFrame(result)

surgs = []
for chunk_idx,chunk in enumerate(pd.read_csv(pj(data_path, 'aims_intraopcaseinfo.csv'), chunksize=50000)):
    print(chunk_idx)
    surgs.append(chunk[['mpog_case_id', 'aims_actual_procedure_text', 'mpog_patient_id', 'aims_patient_age_years']])
surgs = pd.concat(surgs)

demos = []
for chunk_idx,chunk in enumerate(pd.read_csv(pj(data_path, 'aims_patients.csv'), chunksize=50000)):
    print(chunk_idx)
    demos.append(chunk[['mpog_patient_id', 'aims_sex', 'aims_race_text']])
demos = pd.concat(demos)

mort = []
for chunk_idx,chunk in enumerate(pd.read_csv(pj(data_path, 'aims_hospitalmortality.csv'), chunksize=50000)):
    print(chunk_idx)
    mort.append(chunk[['mpog_patient_id', 'date_of_death_range_start']])
mort = pd.concat(mort)

supplement = pd.merge(surgs, demos, on='mpog_patient_id', how='outer')
supplement = pd.merge(mort, supplement, on='mpog_patient_id', how='outer')
supplement = supplement.rename(columns={
                'aims_actual_procedure_text': 'surgery',
                'aims_race_text': 'race',
                'aims_sex': 'sex',
                'aims_patient_age_years': 'age',
                'date_of_death_range_start': 'deathdate',
    })

res = pd.merge(result, supplement, on='mpog_case_id', how='outer')
res.to_csv(pj(data_path,'processed_preop.csv'))
