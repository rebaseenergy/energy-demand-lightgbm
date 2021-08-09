import lightgbm as lgb
import pickle
import sys
import numpy as np
import yaml

in_path = sys.argv[1]
out_path = sys.argv[2]

with open(in_path, 'rb') as f:
    df = pickle.load(f)

params = yaml.safe_load(open('params.yaml'))

# time related features
timestamps = df.index.get_level_values('valid_datetime')
seconds_in_day = 24*60*60
df.loc[:, 'sin_time_hd'] = np.sin(2*np.pi*(timestamps-timestamps.round("D")).total_seconds()/seconds_in_day)
df.loc[:, 'cos_time_hd'] = np.cos(2*np.pi*(timestamps-timestamps.round("D")).total_seconds()/seconds_in_day)
df.loc[:, 'time_hod'] = timestamps.hour


if 'dow' in params['featurize']['features']:
    df.loc[:, 'dow'] = timestamps.dayofweek


df.loc[:, 'cal_weekday'] = timestamps.dayofweek.isin([0, 1, 2, 3, 4]).astype('int')
df.loc[:, 'cal_weekend'] = timestamps.dayofweek.isin([5, 6]).astype('int')

print(df.columns)

df_X = df.drop(columns=['Load(MWh)'])
df_y = df['Load(MWh)']




dataset = lgb.Dataset(df_X, label=df_y, params={'verbose': -1}, free_raw_data=False)



with open(out_path, 'wb') as f:
    pickle.dump(dataset, f)
