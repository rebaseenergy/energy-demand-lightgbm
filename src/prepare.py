import rebase as rb
import sys
import pandas as pd
import yaml
import os
import pickle

rb.api_key = os.environ.get('RB_API_KEY')


in_path = sys.argv[1]
out_path = sys.argv[2]

params = yaml.safe_load(open('params.yaml'))

obs_df = pd.read_csv(in_path)
obs_df['valid_datetime'] = pd.to_datetime(obs_df['Datetime'])

weather_dict = rb.Weather.get(params['prepare']['weather'])

weather_df = pd.DataFrame.from_dict(weather_dict)
weather_df['valid_datetime'] = pd.to_datetime(weather_df['valid_datetime'])


df = weather_df.merge(obs_df, on='valid_datetime')

df.index = pd.MultiIndex.from_arrays(
         [pd.to_datetime(df['ref_datetime'].values),
         pd.to_datetime(df['valid_datetime'].values)],
         names=['ref_datetime', 'valid_datetime'])

df = df.drop(columns=['ref_datetime', 'valid_datetime', 'Datetime'])


with open(out_path, 'wb') as f:
    pickle.dump(df, f)
