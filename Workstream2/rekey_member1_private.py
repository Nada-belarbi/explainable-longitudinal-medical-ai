"""PRIVATE ONLY: convert Member 1 raw NACCID subject keys into Workstream 2 keys.

Do not commit or distribute the source cohort, patient-level data, outputs,
or shared pseudonym secret. This tool is optional and does not fix the missing
anatomical training cohort.
"""
import argparse
import hashlib
import hmac
import json
from pathlib import Path
import pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument('--secret',required=True)
ap.add_argument('--visits',required=True)
ap.add_argument('--cohort',required=True)
ap.add_argument('--splits',required=True)
ap.add_argument('--out',required=True)
a=ap.parse_args()
key=Path(a.secret).read_bytes()
assert len(key)>=32

def pk(x):return 'mri_'+hmac.new(key,str(x).encode('utf8'),hashlib.sha256).hexdigest()[:32]

out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
for input_path, output_name in [(a.visits,'visits_clinical_rekeyed.csv'),(a.cohort,'cohort_manifest_rekeyed.csv')]:
    d=pd.read_csv(input_path,dtype={'NACCID':str})
    if 'NACCID' not in d:raise ValueError('NACCID absent')
    d['subject_key']=d.NACCID.map(pk)
    d=d.drop(columns=[c for c in ['NACCID','NACCMRFI'] if c in d])
    d.to_csv(out/output_name,index=False)

data=json.load(open(a.splits))
if not isinstance(data,list):raise ValueError('Expected original split list')
result=[{'subject_key':pk(item['NACCID']),'split':item['split']} for item in data]
(out/'splits_rekeyed.json').write_text(json.dumps(result,indent=2))
print('Created three RESTRICTED pseudonymized Member 1 files.')
